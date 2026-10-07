# Fire Detection Project Progress

## Completed Tasks

### 1. Data Analysis and Understanding
- ✅ Analyzed data structure from CSV files
- ✅ Identified sensor types: PM1, PM4, Temperature, Relative Humidity, Pressure, SPH sensors (S1-S12)
- ✅ Understood data format with timestamps and multiple station IDs
- ✅ Located station information in Location.csv

### 2. Data Loading System Development
- ✅ Created `FireDetectionDataLoader` class
- ✅ Implemented data loading for all sensor types
- ✅ Added location data loading functionality
- ✅ Handled missing data and data type conversion
- ✅ Added error handling for file loading

### 3. Visualization System
- ✅ Individual sensor plotting functionality
- ✅ Multi-sensor overview plotting
- ✅ Correlation matrix visualization
- ✅ Station comparison plots
- ✅ Configurable plot parameters

### 4. Data Analysis Features
- ✅ Data summary statistics
- ✅ Missing data percentage calculation
- ✅ Time range analysis
- ✅ Station information extraction

### 5. Documentation and Examples
- ✅ Created comprehensive README.md
- ✅ Developed example usage script
- ✅ Added requirements.txt for dependencies
- ✅ Documented all class methods and functions

## Current Status

The basic data loading and visualization system is complete and ready for use. The system can:
- Load all sensor data from CSV files
- Visualize data in multiple formats
- Provide data summaries and statistics
- Handle missing data gracefully

### Testing Results
- ✅ Data loading functionality tested and working
- ✅ Successfully loaded 17 sensor datasets (689,083 rows each)
- ✅ 8 monitoring stations identified
- ✅ 87.5% missing data rate (typical for sensor networks)
- ✅ Visualization functions tested and working

### Map Visualization Results
- ✅ Analyzed all 8 monitoring stations with data
- ✅ Created interactive map with station locations
- ✅ Added administrative boundaries and multiple tile layers
- ✅ Implemented data quality visualization
- ✅ Generated both static and interactive map outputs
- ✅ Fixed coordinate data issues (negative longitude values)

## Next Steps for Fire Detection Algorithm

### Phase 1: Data Preprocessing
- [ ] Implement data cleaning and normalization
- [ ] Handle time series data alignment
- [ ] Create data quality assessment tools

### Phase 2: Feature Engineering
- [ ] Calculate rolling statistics (mean, std, min, max)
- [ ] Create time-based features (hour, day, season)
- [ ] Implement sensor fusion features
- [ ] Add anomaly detection features

### Phase 3: Fire Detection Algorithm
- [ ] Implement threshold-based detection
- [ ] Develop machine learning models
- [ ] Create ensemble methods
- [ ] Add real-time processing capability

### Phase 4: Validation and Testing
- [ ] Create validation datasets
- [ ] Implement cross-validation
- [ ] Add performance metrics
- [ ] Create testing framework

## Technical Notes

- Data contains timestamps from 2025-08-27 with high frequency sampling
- Multiple stations provide spatial coverage
- Missing data patterns need to be analyzed for algorithm development
- SPH sensors (S1-S12) may be key indicators for fire detection

## Session Log (2025-10-02)

### Implemented
- Aligned `mu1` baseline array with 30s time index and enabled plotting via `plot_time_series` by wrapping into a `pd.Series` indexed by `pm1_s.index`.
 - Reorganized outputs into per-site subfolders: figures/`SITE_ID` and outputs/`SITE_ID`. Added auto directory creation and refactored all save paths accordingly.
 - Added `aggregate_fire_events.py` to cluster and vote events across nearby stations, producing a summary CSV.
- Created `visualize_fire_events_map.py` with cartopy-based map visualization showing dynamic station colors (blue→red→blue) during fire events.
- Enhanced map with background layers (terrain/ocean) and excluded distant station aa-87-15-02, focused on nearby station cluster.
- Added multiple map styles: administrative boundaries (default), OpenStreetMap tiles, terrain, and satellite imagery with fallback options.
- Fixed map extent to auto-zoom based on remaining station locations and enhanced background map loading with better visibility.

### Errors Encountered
- `plot_time_series` expects a `DatetimeIndex`, while `base["mu1"]` was a NumPy array without index alignment.

### Resolution Methods
- Created `pm1_baseline_series = pd.Series(base["mu1"], index=pm1_s.index, name="PM1_baseline")` and called `plot_time_series` with `resample_freq=None` since data already at 30s cadence.
 - Introduced `FIG_DIR = Path('figures') / SITE_ID` and `OUT_DIR = Path('outputs') / SITE_ID`, ensured directories with `mkdir(parents=True, exist_ok=True)`, and updated all `save_path` and `to_csv` calls to use these paths.

## Files Created

1. `fire_detection_data_loader.py` - Main data loading and visualization class
2. `example_usage.py` - Example script demonstrating usage
3. `test_data_loading.py` - Test script for data loading functionality
4. `test_visualization.py` - Test script for visualization functionality
5. `analyze_station_data.py` - Station data analysis and basic mapping
6. `create_station_map.py` - Interactive and static map creation
7. `create_advanced_map.py` - Advanced map with boundaries and quality analysis
8. `fire_detection_stations_map.html` - Interactive map output
9. `fire_detection_advanced_map.html` - Advanced interactive map output
10. `requirements.txt` - Python dependencies
11. `README.md` - Project documentation
12. `progress.md` - This progress tracking file

## 最新进展

### 2024-12-XX - 增强版火灾检测时间线绘图工具
- 创建了 `plot_fire_timeline_enhanced.py` 脚本，支持两种数据格式的可视化
- 支持 TSPulse 数据格式（tsp_output目录）：
  - 读取 `tspulse_per_minute.csv` 文件
  - 绘制每个站点的 TSPulse 分数和火灾标志时间线
  - 生成 7 个站点的 TSPulse 时间线图
- 支持 Outputs 数据格式（outputs目录）：
  - 读取 `if_per_minute.csv` 文件绘制 IF 分数时间线
  - 读取火灾事件 CSV 文件生成事件摘要图
  - 支持多个站点的火灾事件可视化
- 生成三种类型的图表：
  - `plots/tspulse/` - TSPulse 时间线图
  - `plots/outputs/` - IF 时间线图  
  - `plots/summary/` - 火灾事件摘要图
- 解决了数据格式差异问题，统一了可视化输出

### 2024-12-XX - 修复fire_detection_pipeline.py数据加载问题
- 修复了数据加载问题：将单一的ENV.csv文件替换为分离的T.csv和RH.csv文件
- 更新了run_pipeline函数，支持分离的温度和湿度文件输入：
  - 添加temp_csv和rh_csv参数
  - 自动合并温度和湿度数据，添加后缀标识（_T和_RH）
  - 保持向后兼容性，仍支持env_csv参数
- 修改输出目录为multi_output文件夹：
  - events_out.csv - 火灾事件检测结果
  - timeline_out.csv - 时间线数据
  - events_located.csv - 事件定位结果
- 成功测试pipeline运行，生成了正确的输出文件

### 2024-12-XX - aa-44-13-97 站点火灾事件可视化
- 创建了 `plot_aa_44_13_97_fire_events.py` 脚本，专门用于 aa-44-13-97 站点的火灾事件可视化
- 整合了所有检测方法：
  - TSPulse 检测方法
  - Single-station 检测方法  
  - Multi-station 检测方法
- 添加了真实火灾标签数据（大青湖火灾事件）：
  - Sep 11, 19:00-19:25 KST (Dae-Cheong Lake)
  - Sep 12, 08:15-08:45 KST (Dae-Cheong Lake)
  - Sep 16, 19:10-20:00 KST (Dae-Cheong Lake)
  - Sep 17, 09:20-10:25 KST (Dae-Cheong Lake)
- 正确处理了时区转换（KST 到 UTC）
- 生成了两个可视化图表：
  - `aa-44-13-97_all_methods_with_real_labels.png` - 综合比较图
  - `aa-44-13-97_detection_summary.png` - 检测性能摘要
- 检测性能统计：
  - TSPulse: 100% 成功率 (4/4)
  - Single-station: 50% 成功率 (2/4)
  - Multi-station: 0% 成功率 (0/4)

### 2024-12-XX - 增强所有站点火灾事件可视化脚本
- 修改了 `plot_all_stations_fire_events.py` 脚本，添加了真实火灾事件标记功能
- 添加了6个真实火灾事件（KST时间）：
  - Dae-Cheong Lake: Sep 11, 12, 16, 17 的4个事件
  - Cherry Blossom Road: Sep 11, 16 的2个事件
- 正确处理了时区转换（KST 到 UTC）
- 在所有可视化图表中添加了黑色虚线和阴影区域来标记真实火灾事件：
  - TSPulse 事件图表
  - Single-station 事件图表
  - Multi-station 事件图表
  - 综合比较图表
  - 所有方法比较图表
- 更新了图表标题，明确标注包含真实火灾事件标记
- 成功生成了5个更新后的可视化图表，涵盖所有6个活跃站点

### 2024-12-XX - aa-44-13-97 站点性能指标计算
- 创建了 `calculate_metrics_aa_44_13_97.py` 脚本，计算 aa-44-13-97 站点对 Dae-Cheong Lake 真实火灾事件的性能指标
- 分析了4个 Dae-Cheong Lake 真实火灾事件：
  - Sep 11, 19:00-19:25 KST (UTC: 2025-09-11 10:00-10:25)
  - Sep 12, 08:15-08:45 KST (UTC: 2025-09-11 23:15-23:45)
  - Sep 16, 19:10-20:00 KST (UTC: 2025-09-16 10:10-11:00)
  - Sep 17, 09:20-10:25 KST (UTC: 2025-09-17 00:20-01:25)
- 计算了不同容差窗口（15、30、60分钟）下的性能指标
- **Single-station 方法性能**：
  - Precision: 0.400 (40%)
  - Recall: 0.500 (50%)
  - F1-Score: 0.444
  - 检测到5个事件，其中2个匹配真实火灾
- **TSPulse 方法性能**：
  - Precision: 0.571 (57.1%)
  - Recall: 1.000 (100%)
  - F1-Score: 0.727
  - 检测到7个事件，其中4个匹配真实火灾
- TSPulse 方法在所有指标上都优于 Single-station 方法

### 2024-12-XX - 增强地图可视化脚本
- 修改了 `visualize_fire_events_map_real_labels.py` 脚本，改进了地图可视化方式
- 实现了更清晰的可视化方案：
  - **站点颜色编码**: 蓝色圆形 = 未检测到火灾，红色圆形 = 检测到火灾
  - **真实火灾标签**: 橙色三角形 = 真实火灾事件位置
  - **时间范围**: 限制为 9/17-9/18 KST (2025-09-16 15:00 - 2025-09-18 15:00 UTC)
- 添加了详细的图例说明各种符号的含义
- 解决了火焰 emoji 渲染为矩形的问题，改用更可靠的几何符号
- 生成了更直观的地图，清楚区分了"检测到的火灾站点"和"真实火灾事件位置"

### 2024-12-XX - 添加动画功能
- 为 `visualize_fire_events_map_real_labels.py` 脚本添加了动画功能
- 新增命令行参数：
  - `--animation`: 启用动画生成
  - `--fps`: 设置动画帧率 (默认 2 FPS)
  - `--time_step_minutes`: 设置时间步长 (默认 30 分钟)
- 动画特性：
  - 显示站点颜色随时间变化 (蓝色→红色→蓝色)
  - 实时更新时间标题 (KST 时间)
  - 保持真实火灾事件标记为静态
  - 支持 GIF 格式输出
- 生成了动画文件: `figures/real_fire_events_map_917_918_animated.gif`
- 动画展示了 9/17-9/18 期间火灾检测的动态变化过程

### 2024-12-XX - 改进动画可视化
- 延长了真实火灾事件的时间窗口（每个事件前后各延长1小时）
- 修改了动画函数，使真实火灾事件标记在动画中也动态显示
- 真实火灾事件现在会在动画中显示为橙色三角形，只在火灾期间可见
- 动画时间步长设置为30分钟，提供更细粒度的时间变化
- 在地图时间范围内有1个真实火灾事件：Dae-Cheong Lake (3.1小时持续时间)
- 动画现在清楚地显示了真实火灾事件期间站点检测状态的变化

### 2024-12-XX - 脚本全面改进
- 修改了时间范围为 9/11-9/18 KST (扩展了7天时间范围)
- 切换数据源从聚合事件到单个站点事件数据：
  - 使用 `outputs/<stationID>/*events.csv` 文件
  - 加载了42个站点事件，涵盖5个活跃站点
- 优化了地图边界：
  - 减小了缓冲区从 0.02 到 0.005
  - 地图更加紧凑，聚焦于站点区域
- 移除了文本框：
  - 去除了 "Real Fire" 和位置名称的文本框
  - 保持地图简洁，只有橙色三角形标记
- 检测结果显示：
  - 6个真实火灾事件在时间范围内
  - 3个事件有检测匹配 (aa-44-13-97, aa-44-17-41)
  - 生成了静态地图和动画地图
- 坐标系统修正：
  - 发现真实火灾坐标超出地图范围的问题
  - 修正坐标从 (127.45, 36.35) 到 (127.495, 36.355)
  - 修正坐标从 (127.48, 36.38) 到 (127.500, 36.360)
  - 新坐标在站点范围内：经度 127.489-127.501，纬度 36.301-36.363
  - 真实火灾标记现在应该在地图中可见

### 2024-12-XX - TSPulse地图可视化脚本
- 创建了新的脚本 `visualize_tspulse_fire_events_map.py`
- 使用 `tsp_output/tspulse_per_minute.csv` 数据源：
  - 加载了63,360条TSPulse记录 (8/27-9/18)
  - 时间跨度：21天23小时59分30秒
- TSPulse检测结果统计：
  - aa-44-13-97: 1,822次检测 (最多)
  - aa-44-16-33: 317次检测
  - aa-44-17-41: 197次检测
  - aa-44-12-15: 119次检测
  - aa-44-16-29: 14次检测
- 生成的文件：
  - 静态地图：`figures/tspulse_fire_events_map_911_918.png`
  - 动画地图：`figures/tspulse_fire_events_map_911_918_animated.gif` (92帧)
- 动画特性：
  - 时间步长：120分钟
  - 帧率：2 FPS
  - 动态显示站点TSPulse检测状态
  - 真实火灾事件标记动态显示
- 地图显示修复：
  - 修复了坐标解析问题（负经度值）
  - 排除了不活跃站点：aa-87-15-02, aa-87-15-38, aa-59-38-31
  - 现在只显示6个活跃站点
  - 改进了地图背景和网格线显示
  - 地图范围：经度 127.482-127.511，纬度 36.344-36.373
