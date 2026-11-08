# G1 可滚动小推车遥操作演示

这是与原 `g1_hand_cart_push_task` 完全独立的 demo task。它引用
`objects/two_wheel_steel_hand_cart_original_rolling.usd`，不修改旧 task、旧控制器或旧资产。

启动模拟器：

```bash
./run_rolling_demo.sh
```

模拟器显示 ready 后，在第二个终端启动 Quest 遥操作：

```bash
./run_quest3s_rolling_demo.sh
```

默认最大履带速度是 `0.18 m/s`，最大加速度是 `0.30 m/s²`。可在启动前覆盖：

```bash
G1_HAND_CART_DEMO_MAX_SPEED_MPS=0.14 G1_HAND_CART_DEMO_MAX_ACCEL_MPS2=0.24 ./run_rolling_demo.sh
```

演示逻辑是有意的运动学合成：小车 root 采用与机器人底盘相同的 XY 位移，左右轮角由左右履带的实际行程按 `θ = s / 0.195455 m` 累积。任务使用 `two_wheel_steel_hand_cart_original_handles_only.usd`：资产根节点是唯一的运动学刚体，只启用左右把手的六个胶囊碰撞体，车斗、车架、支腿、车轴和轮子碰撞关闭；夹爪可以接触和夹住把手，小车的全部视觉部件随同一个根节点同步移动。
