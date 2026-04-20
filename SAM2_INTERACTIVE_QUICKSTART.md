# SAM2 交互标注最短上手清单

适用脚本：
[run_sam2_video_interactive.py](/abs/path/D:/vibe/run_sam2_video_interactive.py:1)

## 1. 一句话理解

- `object_id`：视频里“这个具体目标是谁”
- `class_id`：这个目标“属于哪一类”

最常见的理解方式：

- `object_id` 像“实例编号”
- `class_id` 像“类别编号”

例如：

- 画面里有 2 个人、1 辆车
- 那么你可以这样设：
  - 第一个人：`object_id=1`, `class_id=0`
  - 第二个人：`object_id=2`, `class_id=0`
  - 车：`object_id=3`, `class_id=1`

这里：

- `0` 代表 person
- `1` 代表 car

所以：

- 同类的不同目标，`class_id` 相同，`object_id` 不同
- 同一个目标在所有帧里，`object_id` 要保持不变

## 2. 启动命令

```powershell
D:\miniconda3\envs\vibe\python.exe D:\vibe\run_sam2_video_interactive.py `
  --video D:\vibe\track.mp4 `
  --model-cfg D:\sam2\sam2\sam2\configs\sam2.1\sam2.1_hiera_t.yaml `
  --checkpoint D:\vibe\sam2.1_hiera_tiny.pt `
  --output-dir D:\vibe\sam2_yolo_out `
  --frame-stride 5
```

说明：

- `--frame-stride 1`：每帧都标
- `--frame-stride 5`：每 5 帧保留 1 帧
- 导出的图片和标签会严格同名对应

## 3. 标一个目标的最短流程

1. 在左侧填：
   - `Object ID`
   - `Class ID`
2. 点击 `Set Object/Class`
3. 选择模式：
   - `Positive Point`：正点，点在目标内部
   - `Negative Point`：负点，点在背景或错误区域
   - `Box Drag`：拖一个框
4. 在当前帧上点击或拖框
5. 点击 `Apply Current Prompt`
6. 如果当前帧分割看着没问题，点击 `Propagate Whole Video`
7. 最后点击 `Export YOLO`

## 4. 多个 class 怎么标

你只需要记住一个规则：

- `class_id` 按类别分
- `object_id` 按实例分

### 例子 A：1 个人 + 1 辆车

- person：`object_id=1`, `class_id=0`
- car：`object_id=2`, `class_id=1`

### 例子 B：2 个人 + 1 辆车

- 第一个 person：`object_id=1`, `class_id=0`
- 第二个 person：`object_id=2`, `class_id=0`
- car：`object_id=3`, `class_id=1`

### 例子 C：3 只狗

- 第 1 只狗：`object_id=1`, `class_id=2`
- 第 2 只狗：`object_id=2`, `class_id=2`
- 第 3 只狗：`object_id=3`, `class_id=2`

重点不是编号大小，而是：

- 每个目标一个独立的 `object_id`
- 属于同一类的目标，共享同一个 `class_id`

## 5. 一个完整的多目标操作例子

假设你要标：

- 人 A
- 人 B
- 车 C

并且类别定义是：

- `class_id=0` -> person
- `class_id=1` -> car

那操作顺序可以是：

1. 标人 A
   - `Object ID=1`
   - `Class ID=0`
   - `Set Object/Class`
   - 点正点/框
   - `Apply Current Prompt`

2. 标人 B
   - `Object ID=2`
   - `Class ID=0`
   - `Set Object/Class`
   - 点正点/框
   - `Apply Current Prompt`

3. 标车 C
   - `Object ID=3`
   - `Class ID=1`
   - `Set Object/Class`
   - 点正点/框
   - `Apply Current Prompt`

4. 三个目标都在当前帧看着差不多了
   - 点击 `Propagate Whole Video`

5. 如果某一帧某个目标飘了
   - 切到那一帧
   - 切换回那个目标对应的 `Object ID`
   - 保持它原来的 `Class ID`
   - 补正点/负点
   - `Apply Current Prompt`
   - 再次 `Propagate Whole Video`

6. 全部满意后
   - 点击 `Export YOLO`

## 6. 修正时最容易犯的错

### 错误 1：同一个目标换了新的 `object_id`

错误示例：

- 第 0 帧的人 A 用了 `object_id=1`
- 第 20 帧修正时又写成 `object_id=4`

这样系统会把它当成“新目标”，不是原来的目标。

正确做法：

- 同一个目标从头到尾都用同一个 `object_id`

### 错误 2：把 `class_id` 当成唯一编号

错误示例：

- 画面里 2 个人
- 你都写成：
  - `object_id=0`, `class_id=0`

这样不行，因为两个目标不能共享同一个 `object_id`。

正确做法：

- 人 A：`object_id=1`, `class_id=0`
- 人 B：`object_id=2`, `class_id=0`

### 错误 3：切换对象后忘了点 `Set Object/Class`

建议习惯：

- 每次准备标一个新目标时，都先改 `Object ID` / `Class ID`
- 然后点一次 `Set Object/Class`

## 7. 导出的 YOLO 文件是什么样

每张图片对应一个同名 `.txt`：

- `frames\000000.jpg` 对应 `labels\000000.txt`
- `frames\000005.jpg` 对应 `labels\000005.txt`

每一行格式都是：

```text
class_id x_center y_center width height
```

这里保存的是：

- `class_id`
- 归一化后的矩形框

注意：

- YOLO 标签里不会保存 `object_id`
- `object_id` 只在你交互标注和 SAM2 跟踪时使用
- 导出成 YOLO 后，最终只保留类别和框

这点非常重要：

- `object_id` 是给“跟踪”和“修正同一个实例”用的
- `class_id` 才是最终写进 YOLO 标注里的类别编号

## 8. 最推荐的编号习惯

如果你在做普通检测数据集，推荐这样定：

- `class_id`
  - `0 = person`
  - `1 = car`
  - `2 = bicycle`
  - `3 = dog`

- `object_id`
  - 从 `1` 开始往上加
  - 每出现一个新目标就分配一个新 id

例如当前视频里有：

- 2 个人
- 1 辆车
- 1 条狗

可以写成：

- person A：`object_id=1`, `class_id=0`
- person B：`object_id=2`, `class_id=0`
- car A：`object_id=3`, `class_id=1`
- dog A：`object_id=4`, `class_id=3`

## 9. 最短记忆版

只记这 3 句就够了：

- `object_id`：区分“谁是谁”
- `class_id`：区分“它是什么类”
- 同一个目标跨帧修正时，永远用同一个 `object_id`

