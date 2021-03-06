set DATA=%1

rem Single reference
rem   Timesteps 0,6 (first, last)
rem     Domain 1
rem       Timestep 0
python wats\plots.py compute --stats-dir stats_single_ref_d01_t0 --filter d01 --time-idx 0 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t0 --plots-dir plots_single_ref_d01_t0  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t0 --plots-dir plots_single_ref_d01_t0_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t0 --plots-dir plots_single_ref_d01_t0_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t0 --plots-dir plots_single_ref_d01_t0_Linux --filter Linux  || goto :error
rem       Timestep 6
python wats\plots.py compute --stats-dir stats_single_ref_d01_t6 --filter d01 --time-idx 6 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t6 --plots-dir plots_single_ref_d01_t6  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t6 --plots-dir plots_single_ref_d01_t6_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t6 --plots-dir plots_single_ref_d01_t6_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01_t6 --plots-dir plots_single_ref_d01_t6_Linux --filter Linux  || goto :error
rem   Domain 2
rem       Timestep 0
python wats\plots.py compute --stats-dir stats_single_ref_d02_t0 --filter d02 --time-idx 0 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t0 --plots-dir plots_single_ref_d02_t0  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t0 --plots-dir plots_single_ref_d02_t0_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t0 --plots-dir plots_single_ref_d02_t0_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t0 --plots-dir plots_single_ref_d02_t0_Linux --filter Linux  || goto :error
rem       Timestep 6
python wats\plots.py compute --stats-dir stats_single_ref_d02_t6 --filter d02 --time-idx 6 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t6 --plots-dir plots_single_ref_d02_t6  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t6 --plots-dir plots_single_ref_d02_t6_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t6 --plots-dir plots_single_ref_d02_t6_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02_t6 --plots-dir plots_single_ref_d02_t6_Linux --filter Linux  || goto :error

rem Make vs CMake
rem   Timesteps 0,6 (first, last)
rem     Domain 2
rem       Timestep 0
python wats\plots.py compute --ref-trial-pairs --stats-dir stats_make_cmake_d02_t0 --filter d02 --time-idx 0 ^
%DATA%\wats_Linux_Make_Debug_dmpar ^
%DATA%\wats_Linux_CMake_Debug_dmpar ^
%DATA%\wats_Linux_Make_Debug_dm_sm ^
%DATA%\wats_Linux_CMake_Debug_dm_sm ^
%DATA%\wats_Linux_Make_Debug_serial ^
%DATA%\wats_Linux_CMake_Debug_serial ^
%DATA%\wats_Linux_Make_Debug_smpar ^
%DATA%\wats_Linux_CMake_Debug_smpar ^
%DATA%\wats_Linux_Make_Release_dmpar ^
%DATA%\wats_Linux_CMake_Release_dmpar ^
%DATA%\wats_Linux_Make_Release_dm_sm ^
%DATA%\wats_Linux_CMake_Release_dm_sm ^
%DATA%\wats_Linux_Make_Release_serial ^
%DATA%\wats_Linux_CMake_Release_serial ^
%DATA%\wats_Linux_Make_Release_smpar ^
%DATA%\wats_Linux_CMake_Release_smpar ^
%DATA%\wats_macOS_Make_Debug_dmpar ^
%DATA%\wats_macOS_CMake_Debug_dmpar ^
%DATA%\wats_macOS_Make_Debug_dm_sm ^
%DATA%\wats_macOS_CMake_Debug_dm_sm ^
%DATA%\wats_macOS_Make_Debug_serial ^
%DATA%\wats_macOS_CMake_Debug_serial ^
%DATA%\wats_macOS_Make_Debug_smpar ^
%DATA%\wats_macOS_CMake_Debug_smpar ^
%DATA%\wats_macOS_Make_Release_dmpar ^
%DATA%\wats_macOS_CMake_Release_dmpar ^
%DATA%\wats_macOS_Make_Release_dm_sm ^
%DATA%\wats_macOS_CMake_Release_dm_sm ^
%DATA%\wats_macOS_Make_Release_serial ^
%DATA%\wats_macOS_CMake_Release_serial ^
%DATA%\wats_macOS_Make_Release_smpar ^
%DATA%\wats_macOS_CMake_Release_smpar || goto :error
python wats\plots.py plot --stats-dir stats_make_cmake_d02_t0 --plots-dir plots_make_cmake_d02_t0 || goto :error
rem       Timestep 6
python wats\plots.py compute --ref-trial-pairs --stats-dir stats_make_cmake_d02_t6 --filter d02 --time-idx 6 ^
%DATA%\wats_Linux_Make_Debug_dmpar ^
%DATA%\wats_Linux_CMake_Debug_dmpar ^
%DATA%\wats_Linux_Make_Debug_dm_sm ^
%DATA%\wats_Linux_CMake_Debug_dm_sm ^
%DATA%\wats_Linux_Make_Debug_serial ^
%DATA%\wats_Linux_CMake_Debug_serial ^
%DATA%\wats_Linux_Make_Debug_smpar ^
%DATA%\wats_Linux_CMake_Debug_smpar ^
%DATA%\wats_Linux_Make_Release_dmpar ^
%DATA%\wats_Linux_CMake_Release_dmpar ^
%DATA%\wats_Linux_Make_Release_dm_sm ^
%DATA%\wats_Linux_CMake_Release_dm_sm ^
%DATA%\wats_Linux_Make_Release_serial ^
%DATA%\wats_Linux_CMake_Release_serial ^
%DATA%\wats_Linux_Make_Release_smpar ^
%DATA%\wats_Linux_CMake_Release_smpar ^
%DATA%\wats_macOS_Make_Debug_dmpar ^
%DATA%\wats_macOS_CMake_Debug_dmpar ^
%DATA%\wats_macOS_Make_Debug_dm_sm ^
%DATA%\wats_macOS_CMake_Debug_dm_sm ^
%DATA%\wats_macOS_Make_Debug_serial ^
%DATA%\wats_macOS_CMake_Debug_serial ^
%DATA%\wats_macOS_Make_Debug_smpar ^
%DATA%\wats_macOS_CMake_Debug_smpar ^
%DATA%\wats_macOS_Make_Release_dmpar ^
%DATA%\wats_macOS_CMake_Release_dmpar ^
%DATA%\wats_macOS_Make_Release_dm_sm ^
%DATA%\wats_macOS_CMake_Release_dm_sm ^
%DATA%\wats_macOS_Make_Release_serial ^
%DATA%\wats_macOS_CMake_Release_serial ^
%DATA%\wats_macOS_Make_Release_smpar ^
%DATA%\wats_macOS_CMake_Release_smpar || goto :error
python wats\plots.py plot --stats-dir stats_make_cmake_d02_t6 --plots-dir plots_make_cmake_d02_t6 || goto :error

goto :EOF

:error
exit /b %errorlevel%