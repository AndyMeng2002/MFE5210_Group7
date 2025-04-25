from main_system import env, run_backtest, evaluate_performance, create_gui
import pandas as pd


def run_strategy(
    start_date: str,
    end_date: str,
    mode: str = "backtest",  # or "paper_trading"
    order_method: str = "vwap",  # or "normal", "twap"
    fee_rate: float = 0.0003,
    gui: bool = True
):
    """
    调用my_system接口, 用于执行策略主流程
    参数：
        - start_date (str): 回测开始日期
        - end_date (str): 回测结束日期
        - mode (str): "backtest" 或 "paper_trading"
        - order_method (str): 订单执行方式 ("normal", "twap", "vwap")
        - fee_rate (float): 手续费率
        - gui (bool): 是否显示 GUI 可视化
    """

    # 设置策略参数
    env.strategy_setting.start_date = pd.to_datetime(start_date)
    env.strategy_setting.end_date = pd.to_datetime(end_date)
    env.strategy_setting.mode = mode
    env.strategy_setting.order_method = order_method
    env.strategy_setting.fee_rate = fee_rate

    # 运行主流程
    print(" [1/3] 正在运行回测……")
    run_backtest()

    print("\n[2/3] 回测完成，正在计算绩效指标……")
    evaluate_performance()

    if gui:
        print("\n[3/3] 启动图形界面展示结果……")
        create_gui()


if __name__ == "__main__":
    run_strategy(
        start_date="2024-05-06",
        end_date="2024-11-07",
        mode="backtest",
        order_method="twap",
        gui=True
    )
