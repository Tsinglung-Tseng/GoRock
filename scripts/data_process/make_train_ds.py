from hotpot.simulation.sample import Sample, SampleWithAnger
from hotpot.dataset import AngerDataSet

from hotpot.database import Database
import os
os.environ["DB_CONNECTION"] ="postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] ="postgresql://picluster@192.168.1.96:5432/picluster"

sample_stmt=f"""
    SELECT
	cs.*,
	lmwl.gamma_1_x,
	lmwl.gamma_1_y,
	lmwl.gamma_1_z,
	lmwl.gamma_1_local_x,
	lmwl.gamma_1_local_y,
	lmwl.gamma_1_local_z,
	lmwl.gamma_2_x,
	lmwl.gamma_2_y,
	lmwl.gamma_2_z,
	lmwl.gamma_2_local_x,
	lmwl.gamma_2_local_y,
	lmwl.gamma_2_local_z
FROM
	coincidence_sample cs
	JOIN list_mode_with_local lmwl ON (cs. "eventID" = lmwl. "eventID")
	JOIN experiment_coincidence_event ece ON (cs. "eventID" = ece. "eventID")
WHERE
	ece.experiment_id = 12 limit 32;
    """


sample_df = Database().read_sql(sample_stmt)
swa = SampleWithAnger(sample_df)

agds = AngerDataSet(sample_df)
sipm_counts_n_position = agds.sipm_counts_n_position
anger_infered = agds.anger_infered
source_position = agds.source_position

with open(f'sipm_counts_n_position.npy', 'wb') as f:
    np.save(f, valid_sipm_counts_n_position)

with open(f'anger_infered.npy', 'wb') as f:
    np.save(f, valid_anger_infered)

with open(f'source_position.npy', 'wb') as f:
    np.save(f, valid_source_position)
