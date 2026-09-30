Staged MSSM ColumnFlow submission
=================================

Put these files in the same scripts directory as common_run3_MSSM.sh.

Sample groups:
  data
  backgrounds
  signal       # ggphi + bbphi together
  ggphi
  bbphi

Example for 2022+2023 combined:

1. CalibrateEvents
   ./MSSM_01_CalibrateEvents.sh 22and23_emu data
   ./MSSM_01_CalibrateEvents.sh 22and23_emu backgrounds
   ./MSSM_01_CalibrateEvents.sh 22and23_emu ggphi
   ./MSSM_01_CalibrateEvents.sh 22and23_emu bbphi

2. SelectEvents
   ./MSSM_02_SelectEvents.sh 22and23_emu data
   ./MSSM_02_SelectEvents.sh 22and23_emu backgrounds
   ./MSSM_02_SelectEvents.sh 22and23_emu ggphi
   ./MSSM_02_SelectEvents.sh 22and23_emu bbphi

3. ReduceEvents
   ./MSSM_03_ReduceEvents.sh 22and23_emu data
   ./MSSM_03_ReduceEvents.sh 22and23_emu backgrounds
   ./MSSM_03_ReduceEvents.sh 22and23_emu ggphi
   ./MSSM_03_ReduceEvents.sh 22and23_emu bbphi

4. MergeReducedEvents
   ./MSSM_04_MergeReducedEvents.sh 22and23_emu data
   ./MSSM_04_MergeReducedEvents.sh 22and23_emu backgrounds
   ./MSSM_04_MergeReducedEvents.sh 22and23_emu ggphi
   ./MSSM_04_MergeReducedEvents.sh 22and23_emu bbphi

5. ProduceColumns
   ./MSSM_05_ProduceColumns.sh 22and23_emu data
   ./MSSM_05_ProduceColumns.sh 22and23_emu backgrounds
   ./MSSM_05_ProduceColumns.sh 22and23_emu ggphi
   ./MSSM_05_ProduceColumns.sh 22and23_emu bbphi

6. CreateHistograms
   ./MSSM_06_CreateHistograms.sh 22and23_emu data
   ./MSSM_06_CreateHistograms.sh 22and23_emu backgrounds
   ./MSSM_06_CreateHistograms.sh 22and23_emu ggphi
   ./MSSM_06_CreateHistograms.sh 22and23_emu bbphi

7. MergeHistograms
   ./MSSM_07_MergeHistograms.sh 22and23_emu data
   ./MSSM_07_MergeHistograms.sh 22and23_emu backgrounds
   ./MSSM_07_MergeHistograms.sh 22and23_emu ggphi
   ./MSSM_07_MergeHistograms.sh 22and23_emu bbphi

8. MergeShiftedHistograms
   Data is intentionally skipped here; data only needs nominal MergeHistograms.
   ./MSSM_08_MergeShiftedHistograms.sh 22and23_emu backgrounds
   ./MSSM_08_MergeShiftedHistograms.sh 22and23_emu ggphi
   ./MSSM_08_MergeShiftedHistograms.sh 22and23_emu bbphi

You can run ggphi+bbphi in one submission with:
   ./MSSM_01_CalibrateEvents.sh 22and23_emu signal

and likewise for all following stages.

Additional LAW options can be appended, for example:
   ./MSSM_01_CalibrateEvents.sh 22and23_emu backgrounds --workers 20
