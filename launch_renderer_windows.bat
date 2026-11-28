@echo off
setlocal
rem Optional dedicated renderer; no changes to the normal browser profile.
set "ORIGIN=%~1"
if not defined ORIGIN set "ORIGIN=http://127.0.0.1:8188"
if "%ORIGIN:~-1%"=="/" set "ORIGIN=%ORIGIN:~0,-1%"
pushd "%~dp0" || exit /b 1
for %%I in ("%CD%") do set "EXTENSION_FOLDER=%%~nxI"
popd
set "BROWSER="
set "KIND=Chrome"
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set "BROWSER=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER if exist "%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe" set "BROWSER=%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
if not defined BROWSER set "KIND=Edge"
if not defined BROWSER if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "BROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined BROWSER (
  echo Chrome or Edge was not found. Open the normal dedicated renderer instead.
  pause
  exit /b 1
)
set "PROFILE=%LOCALAPPDATA%\SHARP-VR-Video\RendererProfile\%KIND%"
set "URL=%ORIGIN%/extensions/%EXTENSION_FOLDER%/viewer_gaussian.html?batch=1&worker=1&v=sharp-projection-29"
echo Starting dedicated SHARP renderer in %KIND%. Keep the window open and visible.
start "SHARP renderer" "%BROWSER%" --user-data-dir="%PROFILE%" --no-first-run --no-default-browser-check --disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --disable-features=CalculateNativeWinOcclusion --new-window "%URL%"
endlocal
