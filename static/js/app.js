const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingState = document.getElementById('loadingState');
const resultsContainer = document.getElementById('resultsContainer');
const resultsList = document.getElementById('resultsList');
const resultsCount = document.getElementById('resultsCount');
const noResultsState = document.getElementById('noResultsState');
const noResultsQuery = document.getElementById('noResultsQuery');
const errorState = document.getElementById('errorState');
const errorMessage = document.getElementById('errorMessage');

let currentQuery = '';
let currentPage = 1;

function toggleClearBtn() {
    if (searchInput.value.trim()) {
        clearBtn.classList.remove('hidden');
        clearBtn.classList.add('flex');
    } else {
        clearBtn.classList.add('hidden');
        clearBtn.classList.remove('flex');
    }
}

function clearSearch() {
    searchInput.value = '';
    currentQuery = '';
    currentPage = 1;
    toggleClearBtn();
    resultsContainer.classList.add('hidden');
    noResultsState.classList.add('hidden');
    errorState.classList.add('hidden');
    searchInput.focus();
}

searchInput.addEventListener('input', toggleClearBtn);
clearBtn.addEventListener('click', clearSearch);

async function searchPapers(query, page = 1) {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&page=${page}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || 'Server error');
    return data;
}

function showLoading() {
    loadingState.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    noResultsState.classList.add('hidden');
    errorState.classList.add('hidden');
    searchBtn.disabled = true;
    searchBtn.innerHTML = `
        <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
    `;
}

function hideLoading() {
    loadingState.classList.add('hidden');
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
}

function showResults(data) {
    resultsContainer.classList.remove('hidden');
    resultsCount.textContent = `${data.total} papers (page ${data.page} of ${data.total_pages})`;

    let html = data.results.map((paper, idx) => createPaperCard(paper, idx)).join('');
    html += createPagination(data);

    resultsList.innerHTML = html;
}

function showNoResults(query) {
    noResultsState.classList.remove('hidden');
    noResultsQuery.textContent = `Try different keywords for "${query}"`;
}

function showError(message) {
    errorState.classList.remove('hidden');
    errorMessage.textContent = message;
}

function createPaperCard(paper, index) {
    const abstract = paper.abstract.length > 280
        ? paper.abstract.substring(0, 280) + '...'
        : paper.abstract;

    return `
        <div class="paper-card bg-white rounded-xl border border-slate-200 p-6 hover:shadow-lg hover:border-primary-200 transition-all cursor-pointer" style="animation-delay: ${index * 50}ms" onclick="openPaperDetail('${paper.paper_id}', ${JSON.stringify(paper).replace(/"/g, '&quot;')})">
            <div class="flex items-start justify-between gap-4 mb-3">
                <h3 class="text-base font-semibold text-slate-800 leading-snug flex-1 hover:text-primary-600">${paper.title}</h3>
                <span class="text-xs font-mono text-slate-400 bg-slate-100 px-2 py-1 rounded shrink-0">${paper.paper_id}</span>
            </div>
            <p class="text-sm text-slate-600 leading-relaxed mb-4">${abstract}</p>
            <div class="flex flex-wrap gap-3">
                <div class="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-primary-50 to-primary-100/50 rounded-lg">
                    <div class="w-2 h-2 bg-primary-500 rounded-full"></div>
                    <span class="text-xs text-slate-500">Score</span>
                    <span class="text-sm font-semibold text-primary-600">${paper.score.toFixed(4)}</span>
                </div>
                <div class="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-emerald-50 to-emerald-100/50 rounded-lg">
                    <div class="w-2 h-2 bg-emerald-500 rounded-full"></div>
                    <span class="text-xs text-slate-500">Title</span>
                    <span class="text-sm font-semibold text-emerald-600">${paper.title_score.toFixed(4)}</span>
                </div>
                <div class="flex items-center gap-2 px-3 py-2 bg-gradient-to-r from-amber-50 to-amber-100/50 rounded-lg">
                    <div class="w-2 h-2 bg-amber-500 rounded-full"></div>
                    <span class="text-xs text-slate-500">Abstract</span>
                    <span class="text-sm font-semibold text-amber-600">${paper.abstract_score.toFixed(4)}</span>
                </div>
            </div>
        </div>
    `;
}

const paperModal = document.getElementById('paperModal');
const modalContent = document.getElementById('modalContent');

async function openPaperDetail(paperId, paper) {
    paperModal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';

    modalContent.innerHTML = `
        <div class="flex items-center justify-center py-12">
            <div class="w-8 h-8 border-3 border-slate-200 border-t-primary-500 rounded-full animate-spin"></div>
        </div>
    `;

    try {
        const response = await fetch(`/api/paper/${paperId}`);
        const data = await response.json();

        if (data.success) {
            const p = data.paper;
            modalContent.innerHTML = `
                <div class="space-y-6">
                    <div>
                        <div class="flex flex-wrap gap-2 mb-3">
                            <span class="text-xs font-mono text-primary-600 bg-primary-50 px-3 py-1 rounded-full">arXiv: ${p.paper_id}</span>
                            ${p.categories ? `<span class="text-xs text-slate-500 bg-slate-100 px-3 py-1 rounded-full">${p.categories}</span>` : ''}
                        </div>
                        <h2 class="text-2xl font-bold text-slate-800 leading-tight">${p.title}</h2>
                    </div>
                    
                    ${p.authors ? `
                    <div>
                        <h4 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">Authors</h4>
                        <p class="text-slate-700">${p.authors}</p>
                    </div>
                    ` : ''}
                    
                    <div class="flex flex-wrap gap-4">
                        <div class="flex items-center gap-2 px-4 py-3 bg-primary-50 rounded-xl">
                            <span class="text-sm text-slate-500">Score</span>
                            <span class="text-xl font-bold text-primary-600">${paper.score.toFixed(4)}</span>
                        </div>
                        <div class="flex items-center gap-2 px-4 py-3 bg-emerald-50 rounded-xl">
                            <span class="text-sm text-slate-500">Title</span>
                            <span class="text-xl font-bold text-emerald-600">${paper.title_score.toFixed(4)}</span>
                        </div>
                        <div class="flex items-center gap-2 px-4 py-3 bg-amber-50 rounded-xl">
                            <span class="text-sm text-slate-500">Abstract</span>
                            <span class="text-xl font-bold text-amber-600">${paper.abstract_score.toFixed(4)}</span>
                        </div>
                    </div>
                    
                    <div>
                        <h4 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">Abstract</h4>
                        <p class="text-slate-700 leading-relaxed">${p.abstract}</p>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        ${p.update_date ? `<div><span class="text-slate-500">Last Updated:</span> <span class="text-slate-700">${p.update_date}</span></div>` : ''}
                        ${p.submitter ? `<div><span class="text-slate-500">Submitter:</span> <span class="text-slate-700">${p.submitter}</span></div>` : ''}
                        ${p.journal_ref ? `<div><span class="text-slate-500">Journal:</span> <span class="text-slate-700">${p.journal_ref}</span></div>` : ''}
                        ${p.doi ? `<div><span class="text-slate-500">DOI:</span> <span class="text-slate-700">${p.doi}</span></div>` : ''}
                    </div>
                    
                    ${p.comments ? `
                    <div>
                        <h4 class="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-2">Comments</h4>
                        <p class="text-slate-600 text-sm">${p.comments}</p>
                    </div>
                    ` : ''}
                    
                    <div class="pt-4 border-t border-slate-200 flex flex-wrap gap-3">
                        <a href="https://arxiv.org/abs/${p.paper_id}" target="_blank" 
                           class="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition">
                            View on arXiv
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"></path>
                            </svg>
                        </a>
                        <a href="https://arxiv.org/pdf/${p.paper_id}" target="_blank" 
                           class="inline-flex items-center gap-2 px-4 py-2 border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-lg transition">
                            Download PDF
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                        </a>
                    </div>
                </div>
            `;
        } else {
            modalContent.innerHTML = `<p class="text-red-500">Error: ${data.message}</p>`;
        }
    } catch (error) {
        modalContent.innerHTML = `<p class="text-red-500">Error loading paper details</p>`;
    }
}

function closeModal() {
    paperModal.classList.add('hidden');
    document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

function createPagination(data) {
    if (data.total_pages <= 1) return '';

    const currentPage = data.page;
    const totalPages = data.total_pages;
    const maxVisible = 5;

    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);

    if (endPage - startPage + 1 < maxVisible) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    let pageButtons = '';

    if (startPage > 1) {
        pageButtons += `<button onclick="goToPage(1)" class="w-10 h-10 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50 text-slate-700">1</button>`;
        if (startPage > 2) {
            pageButtons += `<span class="px-2 text-slate-400">...</span>`;
        }
    }

    for (let i = startPage; i <= endPage; i++) {
        const isActive = i === currentPage;
        pageButtons += `
            <button 
                onclick="goToPage(${i})" 
                class="w-10 h-10 rounded-lg text-sm font-medium ${isActive ? 'bg-primary-600 text-white' : 'border border-slate-200 hover:bg-slate-50 text-slate-700'}"
            >${i}</button>
        `;
    }

    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            pageButtons += `<span class="px-2 text-slate-400">...</span>`;
        }
        pageButtons += `<button onclick="goToPage(${totalPages})" class="w-10 h-10 rounded-lg border border-slate-200 text-sm font-medium hover:bg-slate-50 text-slate-700">${totalPages}</button>`;
    }

    return `
        <div class="flex items-center justify-center gap-2 mt-8 pt-6 border-t border-slate-200">
            <button 
                onclick="goToPage(${currentPage - 1})" 
                ${!data.has_prev ? 'disabled' : ''}
                class="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium ${data.has_prev ? 'hover:bg-slate-50 text-slate-700' : 'text-slate-300 cursor-not-allowed'}"
            >
                Prev
            </button>
            ${pageButtons}
            <button 
                onclick="goToPage(${currentPage + 1})" 
                ${!data.has_next ? 'disabled' : ''}
                class="px-4 py-2 rounded-lg border border-slate-200 text-sm font-medium ${data.has_next ? 'hover:bg-slate-50 text-slate-700' : 'text-slate-300 cursor-not-allowed'}"
            >
                Next
            </button>
        </div>
    `;
}

async function goToPage(page) {
    if (!currentQuery) return;
    currentPage = page;

    showLoading();

    try {
        const data = await searchPapers(currentQuery, page);
        hideLoading();
        showResults(data);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

searchForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    if (!query) return;

    currentQuery = query;
    currentPage = 1;
    showLoading();

    try {
        const data = await searchPapers(query, 1);
        hideLoading();
        if (data.total > 0) {
            showResults(data);
        } else {
            showNoResults(query);
        }
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
});

function searchQuery(query) {
    searchInput.value = query;
    toggleClearBtn();
    searchForm.dispatchEvent(new Event('submit'));
}

window.addEventListener('load', () => searchInput.focus());
