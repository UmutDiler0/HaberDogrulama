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

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Prepare data
        const formData = new FormData(form);
        const requestData = {
            model_type: formData.get('model_type'),
            title: formData.get('title'),
            text: formData.get('text')
        };

        // UI Loading State
        btnText.textContent = 'Analiz Ediliyor...';
        spinner.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');
        progressFill.style.width = '0%'; // reset bar

        try {
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

            // Display Results
            // Gelen veri örneği: { label: "Sahte", confidence: 95.5, probs: {"Sahte": 95.5, "Gerçek": 4.5} }
            
            const isFake = data.label.toLowerCase().includes("sahte");

            resultBadge.textContent = data.label.toUpperCase();
            resultBadge.className = 'result-badge ' + (isFake ? 'fake' : 'real');
            
            confidenceScore.textContent = `%${data.confidence}`;
            
            progressFill.className = 'progress-fill ' + (isFake ? 'fake-fill' : 'real-fill');
            
            resultContainer.classList.remove('hidden');

            // Animate progress bar slightly after making it visible
            setTimeout(() => {
                progressFill.style.width = `${data.confidence}%`;
            }, 100);

        } catch (error) {
            alert('Sunucuyla bağlantı kurulamadı: ' + error.message);
        } finally {
            // Restore UI
            btnText.textContent = 'Analiz Et';
            spinner.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // --- History Logic & Charts ---
    let pieChartInstance = null;
    let barChartInstance = null;

    async function loadHistory() {
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            
            let fakeCount = 0;
            let realCount = 0;
            const modelCounts = { 'roberta': 0, 'distilbert': 0, 'lr': 0, 'dt': 0 };
            
            data.forEach(row => {
                // Table Populating
                const tr = document.createElement('tr');
                const isFake = row.label.toLowerCase().includes('sahte');
                tr.innerHTML = `
                    <td>${row.timestamp}</td>
                    <td><span class="badge" style="background:#4f46e5;color:white;">${row.model_type.toUpperCase()}</span></td>
                    <td>${row.title.substring(0, 40)}${row.title.length > 40 ? '...' : ''}</td>
                    <td><span class="badge ${isFake ? 'badge-fake' : 'badge-real'}">${row.label}</span></td>
                    <td>%${row.confidence}</td>
                `;
                tbody.appendChild(tr);
                
                // Stats Calculating
                if(isFake) fakeCount++; else realCount++;
                if(modelCounts[row.model_type] !== undefined) modelCounts[row.model_type]++;
            });
            
            renderCharts(fakeCount, realCount, modelCounts);
        } catch (err) {
            console.error("Geçmiş veriler yüklenemedi:", err);
        }
    }
    
    function renderCharts(fake, real, models) {
        Chart.defaults.color = '#94a3b8'; // Dark mode text color
        
        // Pie Chart
        const pieCtx = document.getElementById('pieChart').getContext('2d');
        if(pieChartInstance) pieChartInstance.destroy();
        pieChartInstance = new Chart(pieCtx, {
            type: 'doughnut',
            data: {
                labels: ['Sahte Haber', 'Gerçek Haber'],
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
    }

});
