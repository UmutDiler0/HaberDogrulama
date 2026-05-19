document.addEventListener('DOMContentLoaded', () => {
    
    // --- Navigation Logic ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes
            navBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            
            // Eğer geçmiş sekmesine tıklandıysa verileri yükle
            if (targetId === 'history-section') {
                loadHistory();
            }
        });
    });

    // --- Form Submission Logic ---
    const form = document.getElementById('prediction-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const spinner = analyzeBtn.querySelector('.spinner');
    const resultContainer = document.getElementById('result-container');
    const resultBadge = document.getElementById('result-badge');
    const confidenceScore = document.getElementById('confidence-score');
    const progressFill = document.getElementById('progress-fill');
    const multiResultContainer = document.getElementById('multi-result-container');
    const multiResultsGrid = document.getElementById('multi-results-grid');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Prepare data
        const formData = new FormData(form);
        const requestData = {
            model_type: formData.get('model_type'),
            title: formData.get('title'),
            text: formData.get('text'),
            true_label: formData.get('true_label')
        };

        // UI Loading State
        btnText.textContent = 'Analiz Ediliyor...';
        spinner.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');
        if (multiResultContainer) {
            multiResultContainer.classList.add('hidden');
            multiResultsGrid.innerHTML = '';
        }
        progressFill.style.width = '0%'; // reset bar

        try {
            if (requestData.model_type === 'all') {
                const models = ['roberta', 'distilbert', 'lr', 'dt'];
                const modelNames = {
                    roberta: 'RoBERTa',
                    distilbert: 'DistilBERT',
                    lr: 'Logistic Regression',
                    dt: 'Decision Tree'
                };

                multiResultsGrid.innerHTML = '';

                // Hepsine aynı anda istek gönder (Parallel Requests)
                const promises = models.map(mKey => 
                    fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model_type: mKey,
                            title: requestData.title,
                            text: requestData.text,
                            true_label: requestData.true_label
                        })
                    }).then(async (res) => {
                        const data = await res.json();
                        if (!res.ok) throw new Error(data.error || `${modelNames[mKey]} hatası`);
                        return { mKey, data };
                    })
                );

                // Tüm sonuçların tamamlanmasını paralel olarak bekle
                const resultsList = await Promise.all(promises);
                
                // Sonuç listesini eşleşen anahtarlara göre nesneye dönüştür
                const resultsObj = {};
                resultsList.forEach(item => {
                    resultsObj[item.mKey] = item.data;
                });

                // Kartları tek seferde oluştur ve göster
                const order = ['roberta', 'distilbert', 'lr', 'dt'];
                order.forEach(mKey => {
                    const res = resultsObj[mKey];
                    if (!res) return;
                    
                    const isFake = res.label.toLowerCase().includes('sahte');
                    const cardClass = isFake ? 'model-result-card fake-card' : 'model-result-card real-card';
                    const badgeClass = isFake ? 'model-badge fake' : 'model-badge real';
                    const fillClass = isFake ? 'model-progress-fill fake-fill' : 'model-progress-fill real-fill';

                    const card = document.createElement('div');
                    card.className = cardClass;
                    card.innerHTML = `
                        <div class="model-name-title">${modelNames[mKey]}</div>
                        <div class="${badgeClass}">${res.label.toUpperCase()}</div>
                        <div class="model-progress-container" style="width: 100%;">
                            <div class="model-progress-label">
                                <span>Güven Skoru</span>
                                <span>%${res.confidence}</span>
                            </div>
                            <div class="model-progress-bar">
                                <div class="model-progress-fill ${fillClass}" id="fill-${mKey}" style="width: 0%"></div>
                            </div>
                        </div>
                    `;
                    multiResultsGrid.appendChild(card);
                });

                multiResultContainer.classList.remove('hidden');

                // Güven barlarını doldur
                setTimeout(() => {
                    order.forEach(mKey => {
                        const res = resultsObj[mKey];
                        if (!res) return;
                        const fillEl = document.getElementById(`fill-${mKey}`);
                        if (fillEl) {
                            fillEl.style.width = `${res.confidence}%`;
                        }
                    });
                }, 100);

            } else {
                // Tekli model tahmini (Mevcut mantık)
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                });

                const data = await response.json();

                if (!response.ok) {
                    alert('Hata: ' + (data.error || 'Bilinmeyen bir hata oluştu.'));
                    return;
                }

                const isFake = data.label.toLowerCase().includes("sahte");

                resultBadge.textContent = data.label.toUpperCase();
                resultBadge.className = 'result-badge ' + (isFake ? 'fake' : 'real');
                
                confidenceScore.textContent = `%${data.confidence}`;
                
                progressFill.className = 'progress-fill ' + (isFake ? 'fake-fill' : 'real-fill');
                
                resultContainer.classList.remove('hidden');

                // Güven barını yumuşakça doldur
                setTimeout(() => {
                    progressFill.style.width = `${data.confidence}%`;
                }, 100);
            }

        } catch (error) {
            alert('Sunucuyla bağlantı kurulamadı: ' + error.message);
        } finally {
            // UI Durumunu Geri Al
            btnText.textContent = 'Analiz Et';
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // --- History Logic & Charts ---
    let pieChartInstance = null;
    let barChartInstance = null;
    let robertaChartInstance = null;
    let distilbertChartInstance = null;
    let lrChartInstance = null;
    let dtChartInstance = null;

    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            
            let fakeCount = 0;
            let realCount = 0;
            const modelCounts = { 'roberta': 0, 'distilbert': 0, 'lr': 0, 'dt': 0 };
            
            // Populate Table & basic counts
            data.forEach(row => {
                const tr = document.createElement('tr');
                const isPredictionFake = row.label.toLowerCase().includes('sahte');
                
                let trueLabelBadge = '';
                let matchBadge = '';
                
                if (row.true_label) {
                    const isTrueFake = row.true_label.toLowerCase().includes('sahte');
                    trueLabelBadge = `<span class="badge ${isTrueFake ? 'badge-fake' : 'badge-real'}">${row.true_label}</span>`;
                    
                    const isCorrect = row.label.trim().toUpperCase() === row.true_label.trim().toUpperCase();
                    matchBadge = `<span class="badge ${isCorrect ? 'badge-success' : 'badge-fail'}">${isCorrect ? 'DOĞRU' : 'HATALI'}</span>`;
                } else {
                    trueLabelBadge = `<span class="badge badge-neutral">Belirsiz</span>`;
                    matchBadge = `<span class="badge badge-neutral">Belirsiz</span>`;
                }

                const predictionBadge = `<span class="badge ${isPredictionFake ? 'badge-fake' : 'badge-real'}">${row.label}</span>`;

                tr.innerHTML = `
                    <td>${row.timestamp}</td>
                    <td><span class="badge" style="background:#4f46e5;color:white;">${row.model_type.toUpperCase()}</span></td>
                    <td>${row.title.substring(0, 40)}${row.title.length > 40 ? '...' : ''}</td>
                    <td>${trueLabelBadge}</td>
                    <td>${predictionBadge}</td>
                    <td>%${row.confidence}</td>
                    <td>${matchBadge}</td>
                `;
                tbody.appendChild(tr);
                
                if(isPredictionFake) fakeCount++; else realCount++;
                if(modelCounts[row.model_type] !== undefined) modelCounts[row.model_type]++;
            });
            
            // Calculate chronological running accuracies for each model individually
            const chronologicalData = [...data].reverse();
            const modelSuccess = {
                'roberta': { runningCorrect: 0, runningTotal: 0, timelineAcc: [], timelineLabels: [] },
                'distilbert': { runningCorrect: 0, runningTotal: 0, timelineAcc: [], timelineLabels: [] },
                'lr': { runningCorrect: 0, runningTotal: 0, timelineAcc: [], timelineLabels: [] },
                'dt': { runningCorrect: 0, runningTotal: 0, timelineAcc: [], timelineLabels: [] }
            };

            chronologicalData.forEach(row => {
                const modelKey = row.model_type.toLowerCase();
                if (row.true_label && modelSuccess[modelKey]) {
                    const isCorrect = row.label.trim().toUpperCase() === row.true_label.trim().toUpperCase();
                    const stats = modelSuccess[modelKey];
                    stats.runningTotal++;
                    if (isCorrect) stats.runningCorrect++;
                    
                    const runningAcc = Math.round((stats.runningCorrect / stats.runningTotal) * 100);
                    stats.timelineAcc.push(runningAcc);
                    stats.timelineLabels.push('T' + stats.runningTotal);
                }
            });

            // Fallback for empty data
            const modelsKeys = ['roberta', 'distilbert', 'lr', 'dt'];
            modelsKeys.forEach(mKey => {
                if (modelSuccess[mKey].timelineAcc.length === 0) {
                    modelSuccess[mKey].timelineAcc = [0];
                    modelSuccess[mKey].timelineLabels = ['Test Yok'];
                }
            });
            
            renderCharts(fakeCount, realCount, modelCounts, modelSuccess);
        } catch (err) {
            console.error("Geçmiş veriler yüklenemedi:", err);
        }
    }
    
    function renderCharts(fake, real, models, modelSuccess) {
        Chart.defaults.color = '#94a3b8'; // Dark mode text color
        
        // Pie Chart
        const pieCtx = document.getElementById('pieChart').getContext('2d');
        if(pieChartInstance) pieChartInstance.destroy();
        pieChartInstance = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: ['Sahte Tahmin', 'Gerçek Tahmin'],
                datasets: [{
                    data: [fake, real],
                    backgroundColor: ['#ef4444', '#10b981'],
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
        
        // Bar Chart
        const barCtx = document.getElementById('barChart').getContext('2d');
        if(barChartInstance) barChartInstance.destroy();
        barChartInstance = new Chart(barCtx, {
            type: 'bar',
            data: {
                labels: ['RoBERTa', 'DistilBERT', 'Log Reg', 'Dec Tree'],
                datasets: [{
                    label: 'Kullanım Sayısı',
                    data: [models.roberta, models.distilbert, models.lr, models.dt],
                    backgroundColor: '#6366f1',
                    borderRadius: 6
                }]
            },
            options: { 
                responsive: true, 
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, ticks: { stepSize: 1 } }
                }
            }
        });

        // 4 Separate X-Y Running Accuracy Line Charts
        const chartConfigs = [
            { id: 'robertaChart', key: 'roberta', color: '#a855f7', fill: 'rgba(168, 85, 247, 0.08)', instanceName: 'robertaChartInstance' },
            { id: 'distilbertChart', key: 'distilbert', color: '#6366f1', fill: 'rgba(99, 102, 241, 0.08)', instanceName: 'distilbertChartInstance' },
            { id: 'lrChart', key: 'lr', color: '#10b981', fill: 'rgba(16, 185, 129, 0.08)', instanceName: 'lrChartInstance' },
            { id: 'dtChart', key: 'dt', color: '#f97316', fill: 'rgba(249, 115, 22, 0.08)', instanceName: 'dtChartInstance' }
        ];

        // Global Chart Instances management
        const chartInstances = {
            robertaChartInstance,
            distilbertChartInstance,
            lrChartInstance,
            dtChartInstance
        };

        chartConfigs.forEach(cfg => {
            const canvasEl = document.getElementById(cfg.id);
            if (!canvasEl) return;
            const ctx = canvasEl.getContext('2d');
            
            // Destroy existing instance safely
            if (cfg.instanceName === 'robertaChartInstance' && robertaChartInstance) robertaChartInstance.destroy();
            else if (cfg.instanceName === 'distilbertChartInstance' && distilbertChartInstance) distilbertChartInstance.destroy();
            else if (cfg.instanceName === 'lrChartInstance' && lrChartInstance) lrChartInstance.destroy();
            else if (cfg.instanceName === 'dtChartInstance' && dtChartInstance) dtChartInstance.destroy();
            
            const stats = modelSuccess[cfg.key];
            
            const newChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: stats.timelineLabels,
                    datasets: [{
                        label: 'Başarı Oranı',
                        data: stats.timelineAcc,
                        borderColor: cfg.color,
                        backgroundColor: cfg.fill,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: cfg.color,
                        pointBorderColor: '#fff',
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.03)' },
                            ticks: { color: '#94a3b8', font: { size: 9 } }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: 'rgba(255, 255, 255, 0.03)' },
                            ticks: {
                                color: '#94a3b8',
                                font: { size: 9 },
                                callback: function(value) { return '%' + value; }
                            }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return ' Başarı: %' + context.parsed.y;
                                }
                            }
                        }
                    }
                }
            });

            // Assign new instance reference
            if (cfg.instanceName === 'robertaChartInstance') robertaChartInstance = newChart;
            else if (cfg.instanceName === 'distilbertChartInstance') distilbertChartInstance = newChart;
            else if (cfg.instanceName === 'lrChartInstance') lrChartInstance = newChart;
            else if (cfg.instanceName === 'dtChartInstance') dtChartInstance = newChart;
        });
    }

});
