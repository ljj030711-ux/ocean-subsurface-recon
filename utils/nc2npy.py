"""
nc转npy数据处理
查看nc，npy，npz文件结构、维度、变量、属性、数据范围与样例内容
"""
import xarray as xr
import numpy as np
import pandas as pd
import os

class NCConverter:
    @staticmethod
    def inspect_nc(file_path, var_name=None, sample_fraction=0.05):
        """
        查看 .nc 文件结构、维度、变量、属性，并显示部分数据样例

        Parameters
        ----------
        file_path : str
            nc 文件路径
        sample_fraction : float, 可选
            显示样本比例 (默认 0.05，即显示 5% 数据用于预览)
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
            print(f"📊 数据样例(前{max(1, int(vals.shape[0] * sample_fraction))}):\n{vals[:max(1, int(vals.shape[0] * sample_fraction))]}")

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
        print(f"⏳时间共 {len(vt)} 个点, 起始: {vt[0]}, 结束: {vt[-1]}")
        print("\n⏱ 时间间隔统计:")
        print(diff.describe())
        print("\n最常见的间隔:")
        print(diff.value_counts().head(5))
        ds.close()

    # =====================================================
    @staticmethod
    def merge_and_crop_nc(
        file_list,
        output_path,
        var_name='sla',
        crop_size=None,
        start_date=None,
        end_date=None,
        sort_by_time=True,
        dtype=np.float32,
    ):
        """
        合并多个 nc 文件中的单个变量 → npy 文件，并裁剪空间大小。

        固定使用当前数据中的维度字段：
        - 2D/海表变量: (time, latitude, longitude) -> (T, H, W)
        - 3D/深度变量: (time, depth, latitude, longitude) -> (T, D, H, W)

        Parameters
        ----------
        file_list : list[str]
            nc 文件路径列表。
        output_path : str
            输出 npy 文件路径，由调用方手动传入。
        var_name : str
            要提取的单个变量名，默认为 'sla'。
        crop_size : tuple or None
            裁剪后的 (latitude, longitude) 大小，默认 (64, 64)。
            None 表示不裁剪空间维度。
        start_date : str or None
            可选起始日期，例如 '2019-01-01'。
        end_date : str or None
            可选结束日期，例如 '2023-12-31'。
        sort_by_time : bool
            是否根据每个 nc 文件内的首个 time 自动排序，默认 True。
        dtype : numpy dtype
            输出数组数据类型，默认 np.float32。
        """
        if not file_list:
            raise ValueError("file_list 不能为空")
        if not isinstance(var_name, str):
            raise TypeError("var_name 只支持单个变量名字符串，不支持多变量堆叠")

        def _select_time(ds):
            if start_date or end_date:
                return ds.sel(time=slice(start_date, end_date))
            return ds

        def _ordered_dims(da):
            if 'depth' in da.dims:
                return ('time', 'depth', 'latitude', 'longitude')
            return ('time', 'latitude', 'longitude')

        def _cropped_dataset(ds):
            if crop_size is None:
                return ds
            return ds.isel({
                'latitude': slice(0, crop_size[0]),
                'longitude': slice(0, crop_size[1]),
            })

        file_info = []
        for file in file_list:
            with xr.open_dataset(file) as ds:
                first_time = pd.to_datetime(ds['time'].values[0])
                last_time = pd.to_datetime(ds['time'].values[-1])
                file_info.append((first_time, last_time, file))

        if sort_by_time:
            file_info.sort(key=lambda item: item[0])

        chunks = []
        spatial_shape = None
        output_ndim = None
        for first_time, last_time, file in file_info:
            with xr.open_dataset(file) as ds:
                ds = _select_time(ds)
                if ds.sizes.get('time', 0) == 0:
                    print(f"⏭️ 跳过无目标时间的数据: {file}")
                    continue
                if var_name not in ds:
                    raise KeyError(f"变量 '{var_name}' 在文件 {file} 中不存在")

                ds = _cropped_dataset(ds)
                dims = _ordered_dims(ds[var_name])
                current_shape = tuple(int(ds.sizes[dim]) for dim in dims)
                if spatial_shape is None:
                    spatial_shape = current_shape[1:]
                    output_ndim = len(current_shape)
                elif current_shape[1:] != spatial_shape:
                    raise ValueError(
                        f"文件 {file} 的非时间维形状 {current_shape[1:]} "
                        f"与前序文件 {spatial_shape} 不一致"
                    )
                elif len(current_shape) != output_ndim:
                    raise ValueError(
                        f"文件 {file} 的变量维度数 {len(current_shape)} "
                        f"与前序文件 {output_ndim} 不一致"
                    )

                chunks.append((file, int(ds.sizes['time'])))
                print(
                    f"📂 已登记: {file} | time {str(first_time.date())} 至 "
                    f"{str(last_time.date())} | 选中 {current_shape}"
                )

        if not chunks:
            raise ValueError("没有任何 nc 数据落在指定时间范围内")

        total_time = sum(length for _, length in chunks)
        output_shape = (total_time,) + spatial_shape
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        out = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=dtype,
            shape=output_shape,
        )
        print(f"输出形状 {var_name}: {output_shape}")

        offset = 0
        for file, length in chunks:
            print(f"📂 正在读取并写入: {file}")
            with xr.open_dataset(file) as ds:
                ds = _cropped_dataset(_select_time(ds))
                dims = _ordered_dims(ds[var_name])
                data = ds[var_name].transpose(*dims).values
                out[offset:offset + length] = data.astype(dtype, copy=False)

            offset += length
            out.flush()
            print(f"✅ 已写入时间步: {offset}/{total_time}")

        print(f"✅ 已保存至: {output_path}")
        return output_path

    
    # =====================================================
    @staticmethod
    def inspect_npy(
        npy_path,
        sample_fraction=0.05,
        file_list=None,
        var_name=None,
        start_date=None,
        end_date=None,
        crop_size=None,
        sort_by_time=True,
    ):
        """
        查看 .npy 文件的基本结构、维度信息、数据范围与样例内容。

        可选传入原始 nc 文件列表和变量名，用原始 nc 抽样验证 npy 是否按时间顺序拼接。

        参数:
        ----------
        npy_path : str
            npy 文件路径
        sample_fraction : float, 可选
            显示样本比例 (默认 0.05，即显示 5% 数据用于预览)
        file_list : list[str] or None
            原始 nc 文件路径列表。传入后会用于验证时间顺序。
        var_name : str or None
            原始 nc 中要验证的变量名，例如 'to' 或 'so'。
        start_date : str or None
            与 merge_and_crop_nc 一致的起始日期筛选。
        end_date : str or None
            与 merge_and_crop_nc 一致的结束日期筛选。
        crop_size : tuple or None
            与 merge_and_crop_nc 一致的空间裁剪大小。
        sort_by_time : bool
            是否按 nc 内部 time[0] 排序后验证，默认 True。
        """

        print(f"🔍 正在检查 npy 文件: {npy_path}")
        data = np.load(npy_path, mmap_mode='r')

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
            4: "可能是 (time, depth/channel, lat, lon)",
        }
        print(f"维度说明: {dim_info.get(ndim, '未知结构，请人工确认')}")
        print(f"是否含有 NaN: {np.isnan(data).any()}")

        # 取部分样本；sample_fraction <= 0 时只看统计和验证信息，不打印大数组。
        total_size = data.shape[0]
        sample_size = int(total_size * sample_fraction)
        if sample_fraction > 0:
            sample_size = max(1, sample_size)
            sample_data = data[:sample_size]

            print(f"\n📊 数据样例 (前 {sample_size}/{total_size} 个时间步):")
            print(sample_data)
        else:
            print("\n📊 数据样例: 已跳过（sample_fraction <= 0）")

        if file_list is not None or var_name is not None:
            if not file_list or not var_name:
                raise ValueError("验证时间顺序时必须同时传入 file_list 和 var_name")
            NCConverter._verify_npy_time_order(
                data=data,
                file_list=file_list,
                var_name=var_name,
                start_date=start_date,
                end_date=end_date,
                crop_size=crop_size,
                sort_by_time=sort_by_time,
            )

        print("\n✅ 检查完成。")

    # =====================================================
    @staticmethod
    def _verify_npy_time_order(
        data,
        file_list,
        var_name,
        start_date=None,
        end_date=None,
        crop_size=None,
        sort_by_time=True,
    ):
        """
        按 merge_and_crop_nc 的逻辑，用 nc 原始数据抽查 npy 时间拼接顺序。
        """
        def _select_time(ds):
            if start_date or end_date:
                return ds.sel(time=slice(start_date, end_date))
            return ds

        def _ordered_dims(da):
            if 'depth' in da.dims:
                return ('time', 'depth', 'latitude', 'longitude')
            return ('time', 'latitude', 'longitude')

        def _cropped_dataset(ds):
            if crop_size is None:
                return ds
            return ds.isel({
                'latitude': slice(0, crop_size[0]),
                'longitude': slice(0, crop_size[1]),
            })

        file_info = []
        for file in file_list:
            with xr.open_dataset(file) as ds:
                first_time = pd.to_datetime(ds['time'].values[0])
                last_time = pd.to_datetime(ds['time'].values[-1])
                file_info.append((first_time, last_time, file))

        if sort_by_time:
            file_info.sort(key=lambda item: item[0])

        segments = []
        offset = 0
        print("\n📅 时间顺序验证:")
        for first_time, last_time, file in file_info:
            with xr.open_dataset(file) as ds:
                ds = _select_time(ds)
                if ds.sizes.get('time', 0) == 0:
                    continue
                if var_name not in ds:
                    raise KeyError(f"变量 '{var_name}' 在文件 {file} 中不存在")

                ds = _cropped_dataset(ds)
                dims = _ordered_dims(ds[var_name])
                selected_times = pd.to_datetime(ds['time'].values)
                length = int(ds.sizes['time'])
                segment = {
                    "file": file,
                    "start": offset,
                    "end": offset + length,
                    "dims": dims,
                    "first_date": selected_times[0].date(),
                    "last_date": selected_times[-1].date(),
                    "original_first": first_time.date(),
                    "original_last": last_time.date(),
                }
                segments.append(segment)
                print(
                    f"[{segment['start']}:{segment['end']}) "
                    f"{segment['first_date']} -> {segment['last_date']} | "
                    f"{os.path.basename(file)}"
                )
                offset += length

        if not segments:
            raise ValueError("没有任何 nc 数据落在指定时间范围内，无法验证")
        if offset != data.shape[0]:
            print(f"❌ 时间长度不一致: npy={data.shape[0]}, nc筛选后={offset}")
        else:
            print(f"✅ 时间长度一致: {offset}")

        check_indices = {0, data.shape[0] - 1}
        for segment in segments:
            check_indices.add(segment["start"])
            check_indices.add(segment["end"] - 1)
        check_indices = sorted(idx for idx in check_indices if 0 <= idx < data.shape[0])

        all_match = True
        for idx in check_indices:
            segment = next(
                item for item in segments
                if item["start"] <= idx < item["end"]
            )
            local_idx = idx - segment["start"]
            with xr.open_dataset(segment["file"]) as ds:
                ds = _cropped_dataset(_select_time(ds))
                nc_time = pd.to_datetime(ds['time'].values[local_idx]).date()
                nc_slice = (
                    ds[var_name]
                    .transpose(*segment["dims"])
                    .isel(time=local_idx)
                    .values
                )

            npy_slice = data[idx]
            match = np.allclose(npy_slice, nc_slice, equal_nan=True)
            all_match = all_match and match
            icon = "✅" if match else "❌"
            print(f"{icon} idx={idx} date={nc_time} match={match}",npy_slice[:5])

        if all_match and offset == data.shape[0]:
            print("✅ 时间顺序验证通过：抽查点均与原始 nc 匹配")
        else:
            print("❌ 时间顺序或数据内容可能不一致，请检查 file_list、日期范围或输出文件")

    # =====================================================
    @staticmethod
    def inspect_npz(npz_path):
        """
        查看结果 .npz 文件的内部结构与内容摘要
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
    # （2002-2023）
    # T-S标签数据（分辨率：0.125）
    nc_files = [
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777530701999.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777530988158.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777531495623.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777531729049.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777531776081.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777531991835.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777532191144.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777532225605.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777532470701.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777532725368.nc",
        "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/T-S FIELD/cmems_obs-mob_glo_phy_my_0.125deg_P1D-m_1777532927140.nc"
    ]

    # SST（分辨率：0.05）
    # nc_files = [
    #     "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/SST/METOFFICE-GLO-SST-L4-REP-OBS-SST_1778060334440.nc",
    #     "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/SST/METOFFICE-GLO-SST-L4-REP-OBS-SST_1778060387859.nc"
    # ]

    # SSH（分辨率：0.125）
    # nc_files = [
    #     "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/SSH/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_1778061400140.nc",
    # ]

    # # SSS（分辨率：0.125）
    # nc_files = [
    #     "/Users/lijunjie/Documents/上大硕士/data/inversion_data_deepth/SSS/cmems_obs-mob_glo_phy-sss_my_multi_P1D_1778061084903.nc",
    # ]

    output_npy = "/Users/lijunjie/Documents/python/ocean-subsurface-recon/data/raw/T-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy"

    var_name = 'to'  # SST 变量名，SSH 可替换为 'adt'，SSS 可替换为 'sos'

    # 1. 查看单个 nc 文件（可根据需要指定变量名，例如 'so' 对应海表盐度）
    # NCConverter.inspect_nc(nc_files[-2], sample_fraction=1, var_name=var_name)

    # 2. 检查 valid_time 连续性
    # NCConverter.check_valid_time(nc_files[0])

    # 3. 合并多个 nc 文件并裁剪为 npy（这里示例用 sos）
    NCConverter.merge_and_crop_nc(nc_files, output_npy, var_name=var_name)

    # 4. 查看生成的 npy 文件结构与样例
    # NCConverter.inspect_npy(output_npy, sample_fraction=0.16667, file_list=nc_files, var_name = var_name)

    # 5. 查看 npz 文件结构（如果有的话）
    # NCConverter.inspect_npz("/Users/lijunjie/Documents/上大硕士/data/" \
    # "eddy_inversion_results/inversion_results_2023-01-01_2023-12-31_persistence.npz")
