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
