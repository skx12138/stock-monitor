$ws = New-Object -ComObject WScript.Shell
$path = [Environment]::GetFolderPath("Desktop") + "\stock_web.lnk"
$s = $ws.CreateShortcut($path)
$s.TargetPath = "cmd.exe"
$s.Arguments = "/k cd /d C:\Users\Admin\AppData\Roaming\reasonix\global-workspace\stock-monitor && python webapp.py"
$s.WorkingDirectory = "C:\Users\Admin\AppData\Roaming\reasonix\global-workspace\stock-monitor"
$s.Description = "A股量化监控 Web"
$s.IconLocation = "C:\Windows\System32\shell32.dll,13"
$s.Save()
Write-Host "Done! Desktop shortcut: stock_web.lnk"
Write-Host "Double-click to start webapp at http://localhost:5000"
