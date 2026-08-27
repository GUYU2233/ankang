$ErrorActionPreference = "Continue"
$log = Join-Path (Get-Location) "lfs-data-fetch.log"
$attempt = 0

while ($true) {
    $attempt++
    Add-Content $log "[$(Get-Date -Format s)] fetch attempt $attempt"
    git lfs fetch origin 7e9840d --include="data/**" 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 30
}

$count = 0
Get-ChildItem data -Recurse -File | ForEach-Object {
    $file = $_
    if ($file.Length -ge 1024) { return }
    $first = Get-Content -LiteralPath $file.FullName -TotalCount 1 -ErrorAction SilentlyContinue
    if ($first -ne "version https://git-lfs.github.com/spec/v1") { return }

    $tmp = $file.FullName + ".lfs-tmp"
    $command = 'git lfs smudge < "' + $file.FullName + '" > "' + $tmp + '"'
    cmd /d /c $command
    if ($LASTEXITCODE -eq 0 -and (Test-Path $tmp) -and (Get-Item $tmp).Length -gt 1024) {
        Move-Item -Force $tmp $file.FullName
        $count++
    } elseif (Test-Path $tmp) {
        Remove-Item $tmp -Force
    }
}

Add-Content $log "[$(Get-Date -Format s)] materialized=$count"
Write-Output "materialized=$count"
