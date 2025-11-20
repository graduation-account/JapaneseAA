// Get DOM elements
const authorGroupSelect = document.getElementById('author-group');
const textInput = document.getElementById('text-input');
const authorSelect = document.getElementById('author-select');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const urlInput = document.getElementById('url-input');
const urlInputGroup = document.getElementById('url-input-group');
const fetchUrlBtn = document.getElementById('fetch-url-btn');
const authorSelectGroup = document.getElementById('author-select-group');
const inputMethodRadios = document.querySelectorAll('input[name="input-method"]');
const analysisModeRadios = document.querySelectorAll('input[name="analysis-mode"]');

// Handle input method change
inputMethodRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (radio.value === 'url') {
            urlInputGroup.style.display = 'block';
            // URL入力時でも著者選択欄を常に表示
            authorSelectGroup.style.display = 'block';
        } else {
            urlInputGroup.style.display = 'none';
            authorSelectGroup.style.display = 'block';
        }
    });
});

// Handle analysis mode change
analysisModeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        // 分析モードに関係なく、著者選択欄は常に表示
        authorSelectGroup.style.display = 'block';
    });
});

// Handle URL fetch button click
fetchUrlBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    
    if (!url) {
        showError('Please enter a URL');
        return;
    }
    
    setLoading(true);
    hideError();
    
    try {
        const response = await fetch('/api/fetch-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });
        
        const data = await response.json();
        
        if (data.success) {
            textInput.value = data.text;
            showError('Text fetched successfully!', 'success');
        } else {
            showError('Error fetching URL: ' + data.error);
        }
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        setLoading(false);
    }
});

// Handle author group change
authorGroupSelect.addEventListener('change', async () => {
    const authorGroup = authorGroupSelect.value;
    
    try {
        const response = await fetch(`/api/authors?author_group=${encodeURIComponent(authorGroup)}`);
        if (!response.ok) {
            throw new Error('Failed to fetch author list');
        }
        
        const data = await response.json();
        const authors = data.authors;
        
        // Update author selection dropdown
        authorSelect.innerHTML = '';
        authors.forEach(author => {
            const option = document.createElement('option');
            option.value = author;
            option.textContent = author;
            authorSelect.appendChild(option);
        });
        
        // 著者選択欄を常に表示
        authorSelectGroup.style.display = 'block';
    } catch (error) {
        showError('Error occurred while fetching author list: ' + error.message);
    }
});

// Handle predict button click
predictBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    const authorGroup = authorGroupSelect.value;
    const selectedInputMethod = document.querySelector('input[name="input-method"]:checked').value;
    const selectedAnalysisMode = document.querySelector('input[name="analysis-mode"]:checked').value;
    
    // Validation
    if (!text) {
        showError('Please enter text');
        return;
    }
    
    // Update UI
    setLoading(true);
    hideError();
    hideResult();
    
    try {
        let result;
        
        // 分析モードに応じてAPIを呼び出し
        if (selectedAnalysisMode === 'whole') {
            // 従来のAPI（selected_authorが必要）
            const selectedAuthor = authorSelect.value;
            if (!selectedAuthor) {
                showError('Please select an author');
                return;
            }
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    selected_author: selectedAuthor,
                    author_group: authorGroup
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error occurred during prediction');
            }
            
            result = await response.json();
            displayResult(result);
        } else if (selectedAnalysisMode === 'detailed') {
            // 新しい詳細分析API
            const response = await fetch('/api/predict-detailed', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    author_group: authorGroup,
                    analysis_mode: selectedAnalysisMode
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error occurred during prediction');
            }
            
            result = await response.json();
            displayDetailedResult(result, selectedAnalysisMode);
        }
    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        setLoading(false);
    }
});

// Display result
function displayResult(result) {
    // Display basic information
    document.getElementById('result-author-group').textContent = result.author_group;
    document.getElementById('result-selected-author').textContent = result.selected_author;
    document.getElementById('result-similarity').textContent = result.similarity_score + '%';
    
    // Display analysis result
    const analysisDiv = document.getElementById('result-analysis');
    analysisDiv.className = 'result-analysis';
    
    if (result.has_unknown_author) {
        analysisDiv.classList.add('warning');
        analysisDiv.innerHTML = `
            <p>⚠ Possible Unknown Author</p>
            <p style="margin-top: 8px; font-size: 0.9rem;">
                Max Probability: ${result.max_prob_author} (${result.max_prob}%) < Threshold ${result.unknown_threshold}%
            </p>
            <p style="margin-top: 4px; font-size: 0.9rem;">
                The input text may not match any of the trained authors (${result.num_authors} authors).
            </p>
        `;
    } else {
        analysisDiv.classList.add('success');
        analysisDiv.innerHTML = `
            <p>✓ Highest similarity with "${result.max_prob_author}" (${result.max_prob}%)</p>
        `;
    }
    
    // Display prediction probabilities
    const predictionsList = document.getElementById('predictions-list');
    predictionsList.innerHTML = '';
    
    result.predictions.forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        if (pred.is_selected) {
            item.classList.add('selected');
        }
        if (pred.is_max) {
            item.classList.add('max');
        }
        
        const authorDiv = document.createElement('div');
        authorDiv.className = 'prediction-author';
        authorDiv.textContent = pred.author;
        
        const badgesDiv = document.createElement('div');
        badgesDiv.className = 'prediction-badges';
        
        if (pred.is_selected) {
            const badge = document.createElement('span');
            badge.className = 'badge badge-selected';
            badge.textContent = 'Selected';
            badgesDiv.appendChild(badge);
        }
        
        if (pred.is_max) {
            const badge = document.createElement('span');
            badge.className = 'badge badge-max';
            badge.textContent = 'Highest Probability';
            badgesDiv.appendChild(badge);
        }
        
        const probDiv = document.createElement('div');
        probDiv.className = 'prediction-probability';
        probDiv.textContent = pred.probability + '%';
        
        const leftDiv = document.createElement('div');
        leftDiv.style.display = 'flex';
        leftDiv.style.alignItems = 'center';
        leftDiv.appendChild(authorDiv);
        if (badgesDiv.children.length > 0) {
            leftDiv.appendChild(badgesDiv);
        }
        
        item.appendChild(leftDiv);
        item.appendChild(probDiv);
        
        // Add progress bar
        const progressContainer = document.createElement('div');
        progressContainer.className = 'progress-bar-container';
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        progressBar.style.width = pred.probability + '%';
        progressContainer.appendChild(progressBar);
        item.appendChild(progressContainer);
        
        predictionsList.appendChild(item);
    });
    
    // Display result section
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Display detailed result
function displayDetailedResult(result, analysisMode) {
    // 通常の結果表示部分を表示（standard_resultがある場合）
    if (result.detailed && result.detailed.standard_result) {
        displayResult(result.detailed.standard_result);
    }
    
    const detailedResultsSection = document.getElementById('detailed-results-section');
    const detailedResults = document.getElementById('detailed-results');
    detailedResults.innerHTML = ''; // クリア
    
    // 細かい分析結果を表示（detailedモードのみ）
    if (result.detailed && !result.detailed.error) {
        const detailedDiv = document.createElement('div');
        detailedDiv.className = 'detailed-section';
        
        let html = '<h4>Sentence-by-Sentence Analysis</h4>';
        
        if (result.detailed.most_common_author) {
            html += `<p><strong>Most Common Author:</strong> ${result.detailed.most_common_author}</p>`;
        }
        
        if (result.detailed.sentence_results && result.detailed.sentence_results.length > 0) {
            html += '<div class="sentence-results">';
            result.detailed.sentence_results.forEach((sentenceResult, idx) => {
                if (sentenceResult.error) {
                    html += `
                        <div class="sentence-item error">
                            <div class="sentence-header">
                                <span class="sentence-number">Sentence ${sentenceResult.sentence_index}</span>
                            </div>
                            <div class="sentence-text">${sentenceResult.sentence}</div>
                            <div class="sentence-error">Error: ${sentenceResult.error}</div>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="sentence-item">
                            <div class="sentence-header">
                                <span class="sentence-number">Sentence ${sentenceResult.sentence_index}</span>
                                <span class="sentence-author">${sentenceResult.max_author} (${sentenceResult.max_score.toFixed(2)}%)</span>
                            </div>
                            <div class="sentence-text">${sentenceResult.sentence}</div>
                        </div>
                    `;
                }
            });
            html += '</div>';
        }
        
        detailedDiv.innerHTML = html;
        detailedResults.appendChild(detailedDiv);
    }
    
    detailedResultsSection.style.display = 'block';
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error message
function showError(message, type = 'error') {
    errorMessage.textContent = message;
    if (type === 'success') {
        errorMessage.style.background = '#d1fae5';
        errorMessage.style.color = '#065f46';
        errorMessage.style.borderLeftColor = '#10b981';
    } else {
        errorMessage.style.background = '#fee2e2';
        errorMessage.style.color = '#991b1b';
        errorMessage.style.borderLeftColor = '#ef4444';
    }
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error message
function hideError() {
    errorSection.style.display = 'none';
}

// Hide result section
function hideResult() {
    resultSection.style.display = 'none';
    const detailedResults = document.getElementById('detailed-results');
    if (detailedResults) {
        detailedResults.innerHTML = '';
    }
    const detailedResultsSection = document.getElementById('detailed-results-section');
    if (detailedResultsSection) {
        detailedResultsSection.style.display = 'none';
    }
}

// Set loading state
function setLoading(loading) {
    predictBtn.disabled = loading;
    fetchUrlBtn.disabled = loading;
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';
    } else {
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
}

// Execute prediction with Enter key (Ctrl+Enter or Cmd+Enter)
textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        predictBtn.click();
    }
});

