param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Tickers = @()
)

Set-Location -Path $PSScriptRoot

Write-Host "`n[0/3] Verification des prerequis..." -ForegroundColor Cyan

$missingTools = @()
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) { $missingTools += "ollama" }
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) { $missingTools += "python" }

if ($missingTools.Count -gt 0) {
    Write-Host "  -> ERREUR CRITIQUE : Les outils suivants sont introuvables dans le PATH :" -ForegroundColor Red
    foreach ($tool in $missingTools) { Write-Host "     - $tool" -ForegroundColor Red }
    Write-Host "  Veuillez les installer ou les ajouter a vos variables d'environnement." -ForegroundColor Yellow
    exit 1
}

Write-Host "  -> Tous les outils requis sont installes." -ForegroundColor Green

Write-Host "`n[1/3] Verification d'Ollama..." -ForegroundColor Cyan

$ollamaUrl = "http://localhost:11434"
$isOllamaRunning = $false

try {
    $null = Invoke-WebRequest -Uri $ollamaUrl -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    $isOllamaRunning = $true
    Write-Host "  -> Ollama est deja actif." -ForegroundColor Green
} catch {}

if (-not $isOllamaRunning) {
    Write-Host "  -> Ollama non detecte. Demarrage en arriere-plan..." -ForegroundColor Yellow
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden

    Write-Host "  -> Attente du demarrage d'Ollama..." -ForegroundColor Yellow
    $retries = 15
    $ready = $false

    for ($i = 0; $i -lt $retries; $i++) {
        try {
            $null = Invoke-WebRequest -Uri $ollamaUrl -TimeoutSec 1 -UseBasicParsing -ErrorAction Stop
            $ready = $true
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }

    if (-not $ready) {
        Write-Host "  -> ERREUR : Le serveur Ollama n'a pas repondu apres $retries secondes." -ForegroundColor Red
        exit 1
    }
    Write-Host "  -> Ollama est pret." -ForegroundColor Green
}

Write-Host "`n[2/3] Verification du modele deepseek-r1:7b..." -ForegroundColor Cyan

$models = ollama list 2>&1
if ($models -notmatch "deepseek-r1:7b") {
    Write-Host "  -> Modele absent. Telechargement en cours (~4.7 GB)..." -ForegroundColor Yellow
    ollama pull deepseek-r1:7b
} else {
    Write-Host "  -> Modele disponible." -ForegroundColor Green
}

Write-Host "`n[3/3] Lancement du pipeline..." -ForegroundColor Cyan

if ($Tickers.Count -gt 0) {
    $tickerArg = $Tickers -join " "
    Write-Host "  -> Execution avec les tickers : $tickerArg" -ForegroundColor White
    python news_pipeline.py --tickers $tickerArg
} else {
    Write-Host "  -> Execution avec les tickers par defaut : AAPL MSFT GOOGL" -ForegroundColor White
    python news_pipeline.py
}
