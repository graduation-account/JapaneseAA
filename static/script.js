// Get DOM elements
const authorGroupSelect = document.getElementById('author-group');
const textInput = document.getElementById('text-input');
const authorSelect = document.getElementById('author-select');
const predictBtn = document.getElementById('predict-btn');
const resultSection = document.getElementById('result-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');

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
    } catch (error) {
        showError('Error occurred while fetching author list: ' + error.message);
    }
});

// Handle predict button click
predictBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    const selectedAuthor = authorSelect.value;
    const authorGroup = authorGroupSelect.value;
    
    // Validation
    if (!text) {
        showError('Please enter text');
        return;
    }
    
    if (!selectedAuthor) {
        showError('Please select an author');
        return;
    }
    
    // Update UI
    setLoading(true);
    hideError();
    hideResult();
    
    try {
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
        
        const result = await response.json();
        displayResult(result);
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

// Show error message
function showError(message) {
    errorMessage.textContent = message;
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
}

// Set loading state
function setLoading(loading) {
    predictBtn.disabled = loading;
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

