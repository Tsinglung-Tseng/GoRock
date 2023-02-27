# GoRock

GoRock是建模了PET数据采集的DSL。包括了：
- 对PET系统的相关建模：
```from hotpot.simulation.image_system import ImageSystem, AlbiraImageSystem```
- 基于Gate 8.2的物理过程仿真的配置的建模：```from hotpot.simulation.mac import MAC```
- 仿真/真实数据的处理工具：```from hotpot.geometry.system import (
    SipmArray, 
    Hit,
    FuncDataFrame,
    HitsEventIDMapping
)```
- 函数式的数据处理工具：```from hotpot.geometry.primiary import (
    Cartesian3,
    Segment,
    Database, 
    Surface, 
    Trapezoid,
    Box,
    Plane
)```

DSL实现了对像函数式的操作，例如：筛选环差为3，且晶体ID为19～59号上的所有的LOR的操作可以表示为：
```python
ring_error_3_mask = np.logical_or(    
    np.logical_and(
        crystalID_15[:,0]<=19, 
        crystalID_15[:,1]>=59
    ),

    np.logical_and(
        crystalID_15[:,1]<=19, 
        crystalID_15[:,0]>=59
    )
)
```

计算raw_sample_25数据集中，探测器与LOR角度的操作可以表示为：
```
Cartesian3_xyz_distance_nrom((
    raw_sample_25.net_infered_lor_local.hstack() - raw_sample_25.real_lor_local.hstack()
)[
    get_close_xy_border(raw_sample_25.real_lor_local.hstack(), 1)
])
```

