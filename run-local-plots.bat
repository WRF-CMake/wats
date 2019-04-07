set DATA=%1

python wats\plots.py compute --stats-dir stats_single_ref_d01 --filter d01 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01 --plots-dir plots_single_ref_d01  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01 --plots-dir plots_single_ref_d01_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01 --plots-dir plots_single_ref_d01_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d01 --plots-dir plots_single_ref_d01_Linux --filter Linux  || goto :error

python wats\plots.py compute --stats-dir stats_single_ref_d02 --filter d02 %DATA%\wats_Linux_Make_Debug_serial %DATA%\*  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02 --plots-dir plots_single_ref_d02  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02 --plots-dir plots_single_ref_d02_macOS --filter macOS  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02 --plots-dir plots_single_ref_d02_Windows --filter Windows  || goto :error
python wats\plots.py plot --stats-dir stats_single_ref_d02 --plots-dir plots_single_ref_d02_Linux --filter Linux  || goto :error

python wats\plots.py compute --ref-trial-pairs --stats-dir stats_make_cmake_d01 --filter d01 ^
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

python wats\plots.py compute --ref-trial-pairs --stats-dir stats_make_cmake_d02 --filter d02 ^
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

python wats\plots.py plot --stats-dir stats_make_cmake_d01 --plots-dir plots_make_cmake_d01 || goto :error
python wats\plots.py plot --stats-dir stats_make_cmake_d02 --plots-dir plots_make_cmake_d02 || goto :error

goto :EOF

:error
exit /b %errorlevel%