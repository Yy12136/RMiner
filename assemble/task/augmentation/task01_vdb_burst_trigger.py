"""Opt-in one-shot playback for the existing BlenderVdbDust3D renderer.

Legacy continuous playback stays unchanged unless the factory is selected.
The density cache is in task world coordinates; no camera image is modified.
"""
import json
import math
import os
from pathlib import Path
import time
import uuid


class Playback:
    def __init__(self, frames, fps, first=25):
        if not 0 <= first < frames or not 0 < fps <= 30:
            raise ValueError('Invalid sequence timing')
        self.frames, self.fps, self.first = frames, fps, first
        self.started = None
        self.trigger_id = ''

    def trigger(self, now, identity, delay=0.):
        if not isinstance(identity, str) or not identity:
            raise ValueError('trigger must be a nonempty string')
        if not math.isfinite(delay) or not 0 <= delay <= 60:
            raise ValueError('delay must be 0..60 seconds')
        if identity == self.trigger_id:
            return False
        self.trigger_id, self.started = identity, now + delay
        return True

    def stop(self):
        self.started = None

    def sample(self, now):
        if self.started is None:
            return 'idle', 0, 0.
        elapsed = now - self.started
        if elapsed < 0:
            return 'waiting', 0, 0.
        duration = (self.frames - self.first) / self.fps
        if elapsed >= duration:
            return 'done', 0, 0.
        index = min(self.frames-1, self.first + int(elapsed*self.fps))
        # The source cache retains a little dust in its last frame. Fade that
        # remainder over the final second before clearing the renderer.
        gain = min(1., max(0., duration-elapsed))
        return 'playing', index, gain


def install_triggered_playback(base):
    class TriggeredBlenderVdbDust3D(base):
        def __init__(self, table_position, table_yaw=0.):
            super().__init__(table_position, table_yaw)
            from pxr import Sdf
            # Use Flow's native composite pass: the saved Task01 cameras can
            # disable the translucency pass in their per-product settings.
            votes=self.stage.DefinePrim(self.ROOT_PATH+'/renderSettings','FlowRenderSettingsParams')
            render_options=dict(compositeEnabled=True,pathTracingEnabled=False,
                                rayTracedTranslucencyEnabled=False,rayTracedReflectionsEnabled=False)
            for key,value in render_options.items():
                votes.CreateAttribute(key,Sdf.ValueTypeNames.Bool).Set(value)
                self._set_setting('/rtx/flow/'+key,value)
            votes.CreateAttribute('flowEnabled',Sdf.ValueTypeNames.Bool).Set(True)
            # The inherited fire preset's 0.05 is nearly invisible over this
            # sub-metre optical path. Tune attenuation, keeping source voxels.
            attenuation=float(os.environ.get('G1_VDB_BURST_ATTENUATION','6.0'))
            if not math.isfinite(attenuation) or not .05 <= attenuation <= 20:
                raise ValueError('G1_VDB_BURST_ATTENUATION must be .05..20')
            self.stage.GetPrimAtPath(self.ROOT_PATH+'/flowRender/rayMarch').GetAttribute('attenuation').Set(attenuation)
            self._playback = Playback(len(self._vdb_frames), self._vdb_fps,
                                     int(os.environ.get('G1_VDB_BURST_FIRST', '25')))
            self._request = Path(os.environ.get('G1_VDB_BURST_REQUEST', '/tmp/g1_task01_vdb_burst_request.json'))
            self._status = Path(os.environ.get('G1_VDB_BURST_STATUS', '/tmp/g1_task01_vdb_burst_status.json'))
            self._session = uuid.uuid4().hex
            self._seen = self._request.read_text() if self._request.exists() else ''
            self._enabled_attr = self.stage.GetPrimAtPath(self.ROOT_PATH+'/flowEmitterNanoVdb').GetAttribute('enabled')
            self._clear_attr = self.stage.GetPrimAtPath(self.ROOT_PATH+'/flowSimulate').GetAttribute('forceClear')
            self._clear_pending = False
            self._was_playing = False
            self._status_at = -1.
            self._error = ''
            # Preserve the renderer's ramp brightness while moving its hue
            # toward the requested lightly warm grey. This is a Flow
            # colormap, not a transfer of Blender's Principled Volume BSDF.
            self._colormap_rgb_table = [
                tuple((sum(rgb)/3)*factor for factor in (1.,.96,.90))
                for rgb in self._colormap_rgb_table]
            self._enabled_attr.Set(False)
            self._rewrite_colormap(0., 0., transient=True)
            # Override only the new burst mode's global haze. Baseline view
            # remains clear until the local VDB is triggered.
            for key in ['/rtx/raytracing/inscattering/densityMult',
                        '/rtx/fog/fogDistanceDensity','/rtx/fog/fogHeightDensity']:
                self.settings.set_float(key,0.)
            self.update(0.)

        def trigger_burst(self, identity=None, delay=0.):
            identity = identity or uuid.uuid4().hex
            if self._playback.trigger(time.monotonic(), identity, delay):
                self._clear_pending = True

        def stop_burst(self):
            self._playback.stop()
            self._clear_pending = True

        def update(self, dt):
            del dt
            now = time.monotonic()
            try:
                raw = self._request.read_text()
                if raw != self._seen:
                    self._seen = raw
                    req = json.loads(raw)
                    if not isinstance(req,dict) or req.get('session') != self._session:
                        raise ValueError('Request belongs to a different simulator session')
                    if req.get('action','trigger') == 'stop':
                        self.stop_burst()
                    elif req.get('action','trigger') == 'trigger':
                        self.trigger_burst(req['trigger'], float(req.get('delay',0)))
                    else:
                        raise ValueError('Unknown burst action')
                    self._error = ''
            except FileNotFoundError:
                pass
            except (ValueError,TypeError,KeyError,OSError) as exc:
                self._error = str(exc)
            phase,index,gain = self._playback.sample(now)
            playing = phase == 'playing'
            clear = self._clear_pending or (self._was_playing and not playing)
            if self._clear_attr:
                self._clear_attr.Set(clear)
            self._clear_pending = False
            self._enabled_attr.Set(playing)
            self._set_vdb_frame(index)
            self._rewrite_colormap(self._colormap_gain*gain, self._colormap_opacity*gain, transient=True)
            self._was_playing = playing
            if now-self._status_at >= .2:
                self._status.parent.mkdir(parents=True,exist_ok=True)
                temp=self._status.with_suffix('.tmp')
                temp.write_text(json.dumps(dict(session=self._session,wall_time=time.time(),phase=phase,
                    frame=index+1,trigger=self._playback.trigger_id,gain=gain,error=self._error)))
                temp.replace(self._status)
                self._status_at=now

        def close(self):
            try:
                if self._status.exists() and json.loads(self._status.read_text()).get('session') == self._session:
                    self._status.unlink()
            finally:
                super().close()
    return TriggeredBlenderVdbDust3D
