Based on the code snippets and concepts you've provided, here’s a refined usage guide that highlights the DSL’s compositional and modular design:

---

## GoRock DSL: A Comprehensive Guide

### Overview
GoRock’s DSL offers a flexible, composable framework for PET (Positron Emission Tomography) data analysis and visualization, particularly geared toward efficient event filtering, coordinate transformation, and Line of Response (LOR) handling. By simplifying complex operations into modular functions, GoRock enables users to focus on high-level data insights with minimal code.

### Core Functionalities

#### 1. Defining System Parameters and Loading Data

Data can be loaded into GoRock using pre-defined system data classes, making it straightforward to set up PET event samples for various experiments:

```python
from hotpot.data_analysis import CachedSystemData

# Load PET data samples for different scenarios
sample_15mm = CachedSystemData('/path/to/data_15mm.csv')
sample_20mm = CachedSystemData('/path/to/data_20mm.csv')
sample_25mm = CachedSystemData('/path/to/data_25mm.csv')
```

#### 2. Visualizing the PET System and LORs

GoRock allows comprehensive visualization of the PET system and Lines of Response (LORs). This feature enables users to graphically compare real, inferred, and estimated LORs, facilitating visual inspection of data quality and positional accuracy.

```python
import plotly.graph_objects as go

# Filter LOR indices based on specific criteria
indices = sample_15mm.estimated_lor_positions[sample_15mm.predicted_local_positions > 7.5].index[:10]

# Visualize the PET system and various LORs in 3D
fig = go.Figure([
    *image_system.to_plotly(),  # System visualization
    *real_lor_positions[indices].to_plotly_segment(marker=dict(color='gold', size=3)),
    *inferred_lor_positions[indices].to_plotly_segment(marker=dict(color='blue', size=3)),
    *predicted_lor_positions[indices].to_plotly_segment(marker=dict(color='green', size=3)),
    sources[indices].to_plotly(marker=dict(size=4, color='red'))
])
fig.show()
```

#### 3. Filtering Out-of-Bound PET Events

To ensure the validity of PET event data, GoRock provides an efficient filtering mechanism for events with Depth of Interaction (DOI) values beyond acceptable thresholds. This helps in focusing analysis on in-range events only.

```python
import numpy as np

# Filter events by DOI thresholds for each sample
out_of_bound_sample_15 = sample_15mm[np.logical_or(
    sample_15mm.predicted_local_positions > 7.5,
    sample_15mm.predicted_local_positions < -7.5
)]

out_of_bound_sample_20 = sample_20mm[np.logical_or(
    sample_20mm.predicted_local_positions > 10,
    sample_20mm.predicted_local_positions < -10
)]

out_of_bound_sample_25 = sample_25mm[np.logical_or(
    sample_25mm.predicted_local_positions > 12.5,
    sample_25mm.predicted_local_positions < -12.5
)]
```

#### 4. Transforming Between Local and Global Coordinate Systems

Coordinate transformation between the PET detector's local and global coordinate systems is essential for accurate positioning and event comparison. GoRock makes this transformation straightforward, allowing seamless alignment of detected and inferred event coordinates.

```python
import numpy as np

# Stack and sort LOR data in both local and global coordinates
sorted_lor_positions = np.sort(
    np.stack([
        sample.real_lor_local.hstack().y, 
        sample.predicted_lor_local.hstack().y
    ], axis=1), 
    axis=0
)
```

### Key Advantages of GoRock’s High-Level DSL

1. **Composable Functions**  
   GoRock’s DSL is highly modular, allowing for straightforward combination of visualization, filtering, and transformation functions.

2. **Streamlined Data Handling**  
   With tools for efficient data loading, event filtering, and DOI validation, GoRock minimizes processing overhead, focusing on essential data insights.

3. **Integrated Visualization**  
   The DSL integrates with Plotly to enable interactive 3D plotting of PET systems and LORs, enhancing data interpretation through graphical insights.

For more advanced examples and additional functionalities, visit the [GoRock GitHub repository](https://github.com/Tsinglung-Tseng/GoRock).
