param(
    [int]$Limit = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Tickers = @()
)

Set-Location -Path $PSScriptRoot

# ---------------------------------------------------------------------------
# [0/4] Prérequis système
# ---------------------------------------------------------------------------

Write-Host "`n[0/4] Verification des prerequis..." -ForegroundColor Cyan

$missingTools = @()
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) { $missingTools += "python" }

if ($missingTools.Count -gt 0) {
    Write-Host "  -> ERREUR CRITIQUE : Les outils suivants sont introuvables dans le PATH :" -ForegroundColor Red
    foreach ($tool in $missingTools) { Write-Host "     - $tool" -ForegroundColor Red }
    Write-Host "  Veuillez les installer ou les ajouter a vos variables d'environnement." -ForegroundColor Yellow
    exit 1
}

Write-Host "  -> Tous les outils requis sont installes." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [1/4] Vérification du fichier .env et des clés API
# ---------------------------------------------------------------------------

Write-Host "`n[1/4] Verification des cles API..." -ForegroundColor Cyan

$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "  -> ERREUR : Fichier .env introuvable." -ForegroundColor Red
    Write-Host "  Creez un fichier .env a la racine du projet avec le contenu suivant :" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "     GROQ_API_KEY=votre_cle_groq" -ForegroundColor White
    Write-Host "     MISTRAL_API_KEY=votre_cle_mistral" -ForegroundColor White
    Write-Host "     CEREBRAS_API_KEY=votre_cle_cerebras" -ForegroundColor White
    Write-Host ""
    Write-Host "  Au moins une cle est requise pour le debat multi-agent." -ForegroundColor Yellow
    exit 1
}

# Lecture du .env pour vérifier qu'au moins une clé est présente
$envContent = Get-Content $envFile
$hasGroq     = ($envContent | Where-Object { $_ -match "^GROQ_API_KEY=.+" }).Count -gt 0
$hasMistral  = ($envContent | Where-Object { $_ -match "^MISTRAL_API_KEY=.+" }).Count -gt 0
$hasCerebras = ($envContent | Where-Object { $_ -match "^CEREBRAS_API_KEY=.+" }).Count -gt 0

if (-not ($hasGroq -or $hasMistral -or $hasCerebras)) {
    Write-Host "  -> ERREUR : Aucune cle API valide trouvee dans le fichier .env." -ForegroundColor Red
    Write-Host "  Au moins une des cles suivantes est requise : GROQ_API_KEY, MISTRAL_API_KEY, CEREBRAS_API_KEY" -ForegroundColor Yellow
    exit 1
}

$activeProviders = @()
if ($hasCerebras) { $activeProviders += "Cerebras" }
if ($hasGroq)     { $activeProviders += "Groq" }
if ($hasMistral)  { $activeProviders += "Mistral" }

Write-Host "  -> Providers detectes : $($activeProviders -join ', ')" -ForegroundColor Green
Write-Host "  -> Provider actif (priorite) : $($activeProviders[0])" -ForegroundColor Green

# ---------------------------------------------------------------------------
# [2/4] Vérification des dépendances Python
# ---------------------------------------------------------------------------

Write-Host "`n[2/4] Verification des dependances Python..." -ForegroundColor Cyan

$requirementsFile = Join-Path $PSScriptRoot "requirements.txt"

if (Test-Path $requirementsFile) {
    Write-Host "  -> Verification et installation des dependances (pip)..." -ForegroundColor Yellow
    python -m pip install -r $requirementsFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> ERREUR lors de la verification ou de l'installation des dependances." -ForegroundColor Red
        Write-Host "  -> Veuillez verifier le contenu de requirements.txt." -ForegroundColor Yellow
    } else {
        Write-Host "  -> Toutes les dependances sont installees et a jour." -ForegroundColor Green
    }
} else {
    Write-Host "  -> AVERTISSEMENT : requirements.txt introuvable. Verification ignoree." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# [3/4] Lancement de news_pipeline.py (collecte)
# ---------------------------------------------------------------------------

Write-Host "`n[3/4] Lancement de la collecte (news_pipeline.py)..." -ForegroundColor Cyan

$tickerArg = if ($Tickers.Count -gt 0) { $Tickers -join " " } else { "AAPL MSFT GOOGL" }
Write-Host "  -> Tickers : $tickerArg" -ForegroundColor White

$limitArgs = if ($Limit -gt 0) { @("--limit", "$Limit") } else { @() }

# PYTHONPATH plus necessaire grace a `pip install -e .` (cf. ADR-008).
# On garde le set PYTHONPATH au cas ou le mode editable n'a pas ete fait.
$env:PYTHONPATH = $PSScriptRoot

# Invocation via `python -m` : import absolu propre, plus de sys.path.insert.
if ($Tickers.Count -gt 0) {
    python -m src.pipelines.news_pipeline --tickers $Tickers @limitArgs
} else {
    python -m src.pipelines.news_pipeline @limitArgs
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> ERREUR : news_pipeline.py a echoue (code $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Host "  -> Collecte terminee." -ForegroundColor Green

# ---------------------------------------------------------------------------
# [4/4] Lancement de agent_pipeline.py (filtrage + débat multi-agent)
# ---------------------------------------------------------------------------

Write-Host "`n[4/4] Lancement du pipeline agent (agent_pipeline.py)..." -ForegroundColor Cyan
Write-Host "  -> FinBERT + Debat multi-agent AutoGen + Consensus" -ForegroundColor White

$limitArgs = if ($Limit -gt 0) { @("--limit", "$Limit") } else { @() }

if ($Tickers.Count -gt 0) {
    python -m src.pipelines.agent_pipeline --tickers $Tickers @limitArgs
} else {
    python -m src.pipelines.agent_pipeline @limitArgs
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  -> ERREUR : agent_pipeline.py a echoue (code $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

Write-Host "`n  -> Pipeline complet termine avec succes." -ForegroundColor Green
Write-Host "  -> Pour visualiser les resultats : streamlit run dashboard/app.py" -ForegroundColor Cyan