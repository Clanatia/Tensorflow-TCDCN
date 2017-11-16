:: Avoid printing all the comments in the Windows cmd
@echo off

SET UNZIP_EXE=unzip\unzip.exe
SET WGET_EXE=wget\wget.exe

:: Download temporary zip
echo ----- Downloading MTFL -----
SET MTFL_FOLDER=/
SET ZIP_NAME=MTFL.zip
SET ZIP_FULL_PATH=%MTFL_FOLDER%%ZIP_NAME%
%WGET_EXE% -c http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/%ZIP_NAME% -P %MTFL_FOLDER%
echo:

echo ----- Unzipping MTFL -----
%UNZIP_EXE% %ZIP_FULL_PATH%
echo:

echo ----- Deleting Temporary Zip File %ZIP_FULL_PATH% -----
del "%ZIP_FULL_PATH%"

echo ----- MTFL Downloaded and Unzipped -----


