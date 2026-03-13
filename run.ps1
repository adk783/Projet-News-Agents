# Usage : .\run.ps1
# Usage avec tickers custom : .\run.ps1 TSLA NVDA AMZN

param(
    [string[]]$Tickers = @()
)

Write-Host "`n[1/3] Vérification d'Ollama..." -ForegroundColor Cyan

$ollamaActif = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    $ollamaActif = $true
} catch {}

if (-not $ollamaActif) {
    Write-Host "  -> Ollama non détecté. Démarrage en arrière-plan..." -ForegroundColor Yellow
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Write-Host "  -> Attente de 5 secondes..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5

    try {
        Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop | Out-Null
        Write-Host "  -> Ollama prêt." -ForegroundColor Green
    } catch {
        Write-Host "  -> ERREUR : Impossible de démarrer Ollama. Vérifiez l'installation." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  -> Ollama déjà actif." -ForegroundColor Green
}

Write-Host "`n[2/3] Vérification du modèle deepseek-r1:7b..." -ForegroundColor Cyan

$modeles = ollama list 2>&1
if ($modeles -notmatch "deepseek-r1:7b") {
    Write-Host "  -> Modèle absent. Téléchargement en cours (~4.7 GB)..." -ForegroundColor Yellow
    ollama pull deepseek-r1:7b
} else {
    Write-Host "  -> Modèle disponible." -ForegroundColor Green
}

Write-Host "`n[3/3] Lancement du pipeline..." -ForegroundColor Cyan

if ($Tickers.Count -gt 0) {
    $tickerArg = $Tickers -join " "
    Write-Host "  -> Tickers : $tickerArg" -ForegroundColor White
    python news_pipeline.py --tickers $tickerArg
} else {
    Write-Host "  -> Tickers par défaut : AAPL MSFT GOOGL" -ForegroundColor White
    python news_pipeline.py
}
