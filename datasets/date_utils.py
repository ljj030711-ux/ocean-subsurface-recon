"""日期处理工具（从 maths_preprocessing 迁移）。"""

from datetime import datetime, timedelta


def generate_date_list(start_date_str, end_date_str):
    """生成 [start, end] 闭区间日期字符串列表。"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [
        (start + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range((end - start).days + 1)
    ]


def date_to_index(select_day, start_date="2019-01-01", end_date="2023-12-31"):
    """将日期字符串映射到时间轴索引（0-based）。"""
    try:
        datetime.strptime(select_day, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"日期格式错误，需要 YYYY-MM-DD，收到：{select_day}") from exc

    full = generate_date_list(start_date, end_date)
    if select_day not in full:
        raise ValueError(f"日期不在范围 [{start_date}, {end_date}] 内：{select_day}")
    return full.index(select_day)


def generate_month_numbers(start_date_str, total_len):
    """按日步长从起始日期生成每个样本对应的月份编号（1..12）。"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    return [
        (start + timedelta(days=i)).month
        for i in range(int(total_len))
    ]


def indices_until_date(total_len, start_date_str, end_date_str):
    """返回 [start, end] 闭区间内、且不超过 total_len 的时间索引。"""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    if end < start:
        return []
    n = min((end - start).days + 1, int(total_len))
    return list(range(max(n, 0)))
