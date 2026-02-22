"""TrackerPipeline 快速测试。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker.pipeline import SystemConfig, TrackerPipeline


def main():
    """主测试函数。"""
    print("=" * 60)
    print("TrackerPipeline 快速测试")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent.parent / "configs" / "system_config.yaml"
    config = SystemConfig.from_yaml(str(config_path))
    print(f"配置已加载")
    
    # 创建管道
    model_path = Path(__file__).parent.parent / config.checkpoint_path
    
    error_count = [0]
    
    def on_error(message: str):
        error_count[0] += 1
        print(f"[错误 {error_count[0]}] {message}")
    
    pipeline = TrackerPipeline(
        model_path=str(model_path),
        config=config,
        error_callback=on_error,
    )
    print(f"TrackerPipeline 已创建")
    
    # 启动
    print("\n正在启动推理管道...")
    if not pipeline.start():
        print("启动失败")
        return
    
    print("推理管道已启动")
    
    # 运行 5 秒
    print("\n运行 5 秒测试...")
    start_time = time.time()
    frame_count = 0
    valid_count = 0
    
    while time.time() - start_time < 5.0:
        result = pipeline.get_latest_result()
        
        if result is not None:
            frame_count += 1
            if result.valid:
                valid_count += 1
            
            # 每秒打印一次状态
            if frame_count % 30 == 0:
                print(f"  帧数: {frame_count}, 有效: {valid_count}, FPS: {result.fps:.1f}")
        
        time.sleep(0.033)  # ~30 FPS
    
    # 停止
    print("\n正在停止...")
    pipeline.stop()
    print("已停止")
    
    # 统计
    print("\n" + "=" * 60)
    print("测试结果:")
    print(f"  总帧数: {frame_count}")
    print(f"  有效帧数: {valid_count}")
    print(f"  有效率: {valid_count/frame_count*100:.1f}%" if frame_count > 0 else "  有效率: N/A")
    print(f"  错误数: {error_count[0]}")
    print("=" * 60)
    
    if frame_count > 0 and error_count[0] == 0:
        print("\n测试通过！")
    else:
        print("\n测试完成，存在问题")


if __name__ == "__main__":
    main()
