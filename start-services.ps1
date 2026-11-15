# start-services.ps1
Write-Host "[1/3] Starting Ollama server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"

Write-Host "Waiting 10 seconds for Ollama to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "[2/3] Activating virtual environment and starting Python server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\Activate.ps1; python server.py"

Write-Host "Waiting 60 seconds for Python server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 60

Write-Host "[3/3] Starting Ngrok tunnel..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "ngrok http 8000"

Write-Host "All services launched successfully!" -ForegroundColor Green