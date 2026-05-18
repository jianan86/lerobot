# LeRobot 数据加载耗时分析结果

## 背景

分析对象是本地数据集：

```text
/home/jianan/workspace/data/lerobot_0511_depth_video
```

训练时使用了 3 个视觉输入：

- `observation.images.fisheye_rgb`
- `observation.images.depth_camera_rgb`
- `observation.depth.depth_camera`

训练现象是 GPU 明显等待 CPU。初步结论是：瓶颈主要不在 parquet 元数据读取，也不在普通 RGB 视频解码，而是在深度视频 `observation.depth.depth_camera` 的逐样本随机读取。

## 数据集概况

从 `meta/info.json` 看到：

- 总帧数：`46692`
- FPS：`30`
- RGB 视频：
  - `observation.images.depth_camera_rgb`
  - `observation.images.fisheye_rgb`
  - 编码：`av1`
  - 分辨率：`480 x 640`
- 深度视频：
  - `observation.depth.depth_camera`
  - 编码：`ffv1`
  - 像素格式：`gray16le`
  - 分辨率：`480 x 640`

视频文件大小：

| 模态 | 大小 |
| --- | ---: |
| `observation.images.fisheye_rgb` | `897M` |
| `observation.images.depth_camera_rgb` | `345M` |
| `observation.depth.depth_camera` | `3.0G` |

## 关键耗时结果

单样本、单帧解码测试结果：

| feature | shape | 耗时 |
| --- | --- | ---: |
| `observation.images.depth_camera_rgb` | `(1, 3, 480, 640)` | `51.45 ms` |
| `observation.images.fisheye_rgb` | `(1, 3, 480, 640)` | `5.60 ms` |
| `observation.depth.depth_camera` | `(1, 1, 480, 640)` | `14630.68 ms` |

RGB-only DataLoader 对照：

| 配置 | 吞吐 |
| --- | ---: |
| `num_workers=0`, `batch_size=8` | 约 `94 fps` |
| `num_workers=4`, `batch_size=8` | 约 `406 fps` |

这说明两个 RGB 视频本身不是主要瓶颈；加入深度视频后，单个样本的深度读取已经达到秒级，足以让 GPU 长时间等待 CPU。

## 代码路径分析

训练读取路径：

```text
LeRobotDataset.__getitem__
  -> DatasetReader.get_item
    -> _get_current_item
    -> _get_query_timestamps
    -> _query_videos
      -> decode_video_frames              # RGB video
      -> decode_depth_video_frames        # depth video
```

其中 RGB 视频走 `decode_video_frames`，默认可使用 `torchcodec` decoder cache。

深度视频走 `decode_depth_video_frames`，当前实现会：

1. 每次调用都 `av.open(video_path, "r")`
2. 从头顺序 `container.decode(stream)`
3. 把整段 depth video 的所有帧转成 `gray16le`
4. 再用时间戳距离挑出目标帧

因此对于随机采样训练，每个样本都可能重复解码一个很大的 `.mkv` 文件。这个行为和 GPU 侧等待 CPU 的现象一致。

## 复现实验命令

新增的 profiling 脚本：

```bash
uv run python benchmarks/dataset/run_dataset_loading_benchmark.py \
  --repo-id local/lerobot_0511_depth_video \
  --root /home/jianan/workspace/data/lerobot_0511_depth_video \
  --tolerance-s 0.02 \
  --return-uint8 \
  --variants metadata rgb depth full \
  --num-samples 8 \
  --batch-size 8 \
  --num-workers 0 4 \
  --csv /tmp/lerobot_dataset_loading.csv
```

如果只想快速确认 RGB 路径是否正常：

```bash
uv run python benchmarks/dataset/run_dataset_loading_benchmark.py \
  --repo-id local/lerobot_0511_depth_video \
  --root /home/jianan/workspace/data/lerobot_0511_depth_video \
  --tolerance-s 0.02 \
  --return-uint8 \
  --variants metadata rgb \
  --num-samples 2 \
  --batch-size 4 \
  --num-workers 0 4 \
  --measure-batches 2
```

注意：`tolerance_s=1e-4` 时 depth 路径曾出现约 `0.0007s` 的时间戳偏差错误。为了分析加载耗时，上面的命令使用 `--tolerance-s 0.02`。

## 结论

当前最耗时步骤是：

```text
decode_depth_video_frames(observation.depth.depth_camera)
```

主要原因不是 `num_workers` 太小，而是 depth video 的随机取帧实现会重复完整解码大文件。继续增加 `num_workers` 只能并行更多 CPU 解码任务，可能缓解一点吞吐，但也会显著增加 CPU 和 IO 压力，不能从根本上解决问题。

## 优化方向

优先级建议：

1. 避免训练时使用当前 FFV1 depth video 随机解码路径。
2. 将 depth 数据改成更适合随机访问的存储格式，例如逐帧 `depth_image`、chunked tensor、zarr、webdataset shard，或可直接按帧索引读取的格式。
3. 如果继续使用 depth video，需要实现 depth decoder cache 和 seek-based frame access，避免每次从头解完整段视频。
4. 对训练配置做临时验证时，可以先只使用两个 RGB key，确认 GPU 利用率是否恢复。
5. 若 policy 需要多帧 observation delta window，depth 解码成本会按查询帧数进一步放大，需要单独测 `--observation-delta-indices` 的影响。

