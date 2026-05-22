# 在 GitHub 网页创建空仓库后执行本脚本推送
# 仓库地址: https://github.com/Super-LiAo808/langchainRagasSmith
# 创建步骤: https://github.com/new → Repository name: langchainRagasSmith → 不要勾选 README

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".git")) {
    Write-Error "未找到 .git，请先在项目根目录执行 git init"
}

git branch -M main 2>$null
$remote = git remote get-url origin 2>$null
if (-not $remote) {
    git remote add origin "https://github.com/Super-LiAo808/langchainRagasSmith.git"
}

Write-Host "推送到 origin main ..."
git push -u origin main
Write-Host "完成: https://github.com/Super-LiAo808/langchainRagasSmith"
