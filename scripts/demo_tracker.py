"""TrackerPipeline 演示脚本。

展示如何使用 TrackerPipeline 进行实时视线追踪。
按 'q' 键退出。
"""
import sys
from pathlib import Path



import cv2

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracker.pipeline import SystemConfig, TrackerPipeline


def main():
    """主函数。"""
    # 加载配置
    config_path = Path(__file__).parent.parent / "configs" / "system_config.yaml"
    
    if config_path.exists():
        config = SystemConfig.from_yaml(str(config_path))
        print(f"已加载配置: {config_path}")
    else:
        config = SystemConfig()
        print("使用默认配置")
    
    # 错误回调
    def on_error(message: str):
        print(f"[错误] {message}")
    
    # 创建 TrackerPipeline
    model_path = Path(__file__).parent.parent / config.checkpoint_path
    
    pipeline = TrackerPipeline(
        model_path=str(model_path),
        config=config,
        error_callback=on_error,
    )
    
    # 启动推理管道
    print("正在启动推理管道...")
    if not pipeline.start():
        print("启动失败，退出")
        return
    
    print("推理管道已启动，按 'q' 键退出")
    print("=" * 60)
    
    try:
        while True:
            # 获取最新结果
            result = pipeline.get_latest_result()
            frame = pipeline.get_latest_frame()
            
            if result is not None:
                # 打印性能指标
                print(f"\rFPS: {result.fps:.1f} | ", end="")
                
                if result.valid and result.gaze_point is not None:
                    px, py = result.gaze_point
                    print(f"注视点: ({px:.0f}, {py:.0f}) | ", end="")
                    
                    # 显示各阶段耗时
                    total_time = sum(result.timings.values())
                    print(f"总耗时: {total_time:.1f}ms", end="")
                else:
                    if result.error_message:
                        print(f"状态: {result.error_message}", end="")
                    else:
                        print("状态: 无效", end="")
            
            # 显示摄像头画面（可选）
            if frame is not None:
                display_frame = frame.copy()
                
                # 叠加注视点
                if result is not None and result.valid and result.gaze_point is not None:
                    # 注意：这里显示的是相对于屏幕的注视点，不是相对于摄像头画面
                    # 仅作为状态指示
                    h, w = display_frame.shape[:2]
                    cv2.putText(
                        display_frame,
                        f"Gaze: ({result.gaze_point[0]:.0f}, {result.gaze_point[1]:.0f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    
                    if result.fps > 0:
                        cv2.putText(
                            display_frame,
                            f"FPS: {result.fps:.1f}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                
                cv2.imshow("Tracker Demo", display_frame)
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    finally:
        # 停止推理管道
        print("\n正在停止推理管道...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
