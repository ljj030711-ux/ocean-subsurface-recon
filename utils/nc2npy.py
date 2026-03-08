import xarray as xr
import numpy as np
import pandas as pd
import os

class NCConverter:
    @staticmethod
    def inspect_nc(file_path, var_name=None):
        """
        查看 .nc 文件结构、维度、变量、属性，并显示部分数据样例

        Parameters
        ----------
        file_path : str
            nc 文件路径
        var_name : str or None
            若指定则只显示该变量的信息
        """
        print(f"🔍 正在读取文件: {file_path}")
        ds = xr.open_dataset(file_path, engine="netcdf4")

        print("\n📂 === 文件概览 ===")
        print(ds)

        # 输出维度信息
        print("\n📏 === 维度信息 ===")
        for name, dim in ds.dims.items():
            print(f"{name}: {dim}")

        # 输出变量信息
        print("\n🔢 === 变量信息 ===")
        vars_to_show = [var_name] if var_name else list(ds.data_vars)
        for var in vars_to_show:
            if var not in ds.data_vars:
                print(f"⚠️ 变量 '{var}' 不在文件中")
                continue
            print(f"\n变量名: {var}")
            print(f"维度: {ds[var].dims}")
            print(f"形状: {ds[var].shape}")
            print(f"属性: {ds[var].attrs}")
            vals = ds[var].values
            print(f"📊 数据样例(前1/3):\n{vals[:max(1, vals.shape[0]//3)]}")

        # 输出经纬度
        print("\n🌍 经纬度范围:")
        if 'latitude' in ds:
            print("纬度:", ds['latitude'].values)
        if 'longitude' in ds:
            print("经度:", ds['longitude'].values)

        ds.close()
        print("\n✅ 文件查看完成。")

    # =====================================================
    @staticmethod
    def check_valid_time(file_path):
        """
        检查 （time） 时间间隔是否连续
        """
        ds = xr.open_dataset(file_path)
        if 'time' not in ds:
            print("⚠️ 此文件中没有 time 变量。")
            return
        vt = pd.to_datetime(ds['time'].values)
        diff = vt.to_series().diff().dropna()
        print(f"时间共 {len(vt)} 个点, 起始: {vt[0]}, 结束: {vt[-1]}")
        print("\n⏱ 时间间隔统计:")
        print(diff.describe())
        print("\n最常见的间隔:")
        print(diff.value_counts().head(5))
        ds.close()

    # =====================================================
    @staticmethod
    def merge_and_crop_nc(file_list, output_path, var_name='sla', crop_size=(64, 64)):
        """
        合并多个 nc 文件中指定变量 → npy 文件，并裁剪为指定大小

        Parameters
        ----------
        file_list : list[str]
            nc 文件路径列表（例如2022、2023、2024）
        output_path : str
            输出 npy 文件路径
        var_name : str
            要提取的变量名，默认为 'sla'
        crop_size : tuple
            裁剪后的 (lat, lon) 大小，默认(64, 64)
        """
        arr_list = []

        for file in file_list:
            print(f"📂 正在读取: {file}")
            ds = xr.open_dataset(file)
            if var_name not in ds:
                raise KeyError(f"变量 '{var_name}' 在文件 {file} 中不存在")
            arr_list.append(ds[var_name].values)
            ds.close()

        # 拼接时间维度
        all_data = np.concatenate(arr_list, axis=0)
        print(f"拼接后形状 {var_name}:{all_data.shape}")

        data = all_data

        # 自动裁剪为指定大小（默认删除最后几行几列）
        if data.ndim == 3:
            data_cropped = data[:, :crop_size[0], :crop_size[1]]
        elif data.ndim == 4:
            data_cropped = data[:, :, :crop_size[0], :crop_size[1]]
        else:
            data_cropped = data
        print(f"裁剪后形状: {data_cropped.shape}")

        np.save(output_path, data_cropped)
        print(f"✅ 已保存至: {output_path}")

    # =====================================================
    @staticmethod
    def verify_point(file_path, npy_path, lat_val, lon_val, var_name='sla', time_steps=5):
        """
        验证 nc 文件与 npy 文件在指定经纬度点上的一致性

        Parameters
        ----------
        file_path : str
            原始 nc 文件路径
        npy_path : str
            转换后的 npy 文件路径
        lat_val, lon_val : float
            目标经纬度值
        var_name : str
            要比较的变量名，默认为 'sla'
        time_steps : int
            比较的时间步数
        """
        ds = xr.open_dataset(file_path)
        lat = ds['latitude'].values
        lon = ds['longitude'].values

        # 找索引 (使用最近的匹配，因为经纬度是浮点数)
        lat_idx = np.argmin(np.abs(lat - lat_val))
        lon_idx = np.argmin(np.abs(lon - lon_val))

        print(f"目标纬度 {lat_val}, 找到索引 {lat_idx}, 实际纬度 {lat[lat_idx]}")
        print(f"目标经度 {lon_val}, 找到索引 {lon_idx}, 实际经度 {lon[lon_idx]}")

        # 提取nc中前time_steps数据
        if var_name not in ds:
            raise KeyError(f"变量 '{var_name}' 在文件中不存在")
        nc_arr = ds[var_name].values[:time_steps]
        ds.close()
        # 如果存在额外维度（如depth），压缩掉
        nc_arr = np.squeeze(nc_arr)
        # 现在应为 (time, lat, lon)
        nc_vals = nc_arr[:, lat_idx, lon_idx]

        # 加载npy文件
        npy_data = np.load(npy_path)
        npy_arr = np.squeeze(npy_data[:time_steps])
        npy_vals = npy_arr[:, lat_idx, lon_idx]

        print("\n🔍 对比前5个时间步:")
        print(f"{var_name} (nc):", nc_vals)
        print(f"{var_name} (npy):", npy_vals)

        print("\n✅ 匹配结果:")
        print(f"{var_name} 匹配:", np.allclose(nc_vals, npy_vals))



    @staticmethod
    def inspect_npy(npy_path, sample_fraction=0.05):
        """
        查看 .npy 文件的基本结构、维度信息、数据范围与样例内容

        参数:
        ----------
        npy_path : str
            npy 文件路径
        sample_fraction : float, 可选
            显示样本比例 (默认 0.1，即显示 10% 数据用于预览)
        """

        print(f"🔍 正在检查 npy 文件: {npy_path}")
        data = np.load(npy_path)

        print("\n📦 基本信息:")
        print(f"类型: {type(data)}")
        print(f"数据类型: {data.dtype}")
        print(f"形状: {data.shape}")

        # 判断维度含义
        # 因为.npy文件只保存了原始的NumPy数组数据（即纯数值），但没有任何语义信息。
        # 不像.nc文件那样保存了维度名（latitude、longitude、time等） 和变量名（u100、v100等）
        ndim = data.ndim
        dim_info = {
            2: "可能是 (time, feature)",
            3: "可能是 (time, lat, lon)",
            4: "可能是 (time, var, lat, lon)",
        }
        print(f"维度说明: {dim_info.get(ndim, '未知结构，请人工确认')}")

        print("\n📈 数值统计:")
        print(f"最小值: {np.nanmin(data)}")
        print(f"最大值: {np.nanmax(data)}")
        print(f"均值: {np.nanmean(data):.4f}")
        print(f"标准差: {np.nanstd(data):.4f}")
        print(f"是否含有 NaN: {np.isnan(data).any()}")

        # 取部分样本
        total_size = data.shape[0]
        sample_size = max(1, int(total_size * sample_fraction))
        sample_data = data[:sample_size]

        print(f"\n📊 数据样例 (前 {sample_size}/{total_size} 个时间步):")
        print(sample_data)

        print("\n✅ 检查完成。")

    def inspect_npz(npz_path):
        """
        查看 npz 文件的内部结构与内容摘要
        """
        data = np.load(npz_path)
        print(f"✅ 成功加载文件: {npz_path}")
        print(f"包含的键名 (keys): {data.files}\n")

        # 遍历所有 key
        for key in data.files:
            arr = data[key]
            print(f"🔸 [{key}]")
            print(f"   形状 shape: {arr.shape}")
            print(f"   数据类型 dtype: {arr.dtype}")
            # 查看是否有缺失值
            print(f"   是否含有 NaN: {np.isnan(arr).any()}")

            # 打印前几个值（避免太大）
            flat = arr.flatten()
            sample = flat[:5] if flat.size > 5 else flat
            print(f"   前几个值: {sample}\n")

        print("🎯 完成查看。")

if __name__ == "__main__":
    # # 示例用法
    # nc_files = [
    #     # "data/ssh_2022.nc",
    #     # "data/ssh_2023.nc",
    #     "/Users/lijunjie/Documents/上大硕士/data/inversion_data/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1772876750929.nc",
    # ]
    # output_npy = "data/raw/subsurface.npy"

    # # 1. 查看单个 nc 文件（可根据需要指定变量名，例如 'sos' 对应海表盐度）
    # NCConverter.inspect_nc(nc_files[0], var_name='so')

    # # 2. 检查 valid_time 连续性
    # NCConverter.check_valid_time(nc_files[0])

    # # 3. 合并多个 nc 文件并裁剪为 npy（这里示例用 sos）
    # NCConverter.merge_and_crop_nc(nc_files, output_npy, var_name='so')

    # # 4. 验证 nc 与 npy 在特定经纬度点上的一致性（变量名可指定）
    # NCConverter.verify_point(nc_files[0], output_npy, lat_val=10.0, lon_val=110.0, var_name='so')

    # # 5. 查看生成的 npy 文件结构与样例
    # NCConverter.inspect_npy(output_npy)

    # 6. 查看 npz 文件结构（如果有的话）
    NCConverter.inspect_npz("/Users/lijunjie/Documents/上大硕士/data/" \
    "eddy_inversion_results/inversion_results_2023-01-01_2023-12-31_persistence.npz")

