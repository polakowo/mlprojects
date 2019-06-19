cards = [{
    "img_url": "img/sap-hana.png",
    "title": "OCCUPATIONAL FRAUD DETECTION",
    "description": "Detect patterns of occupational fraud activity in ERP systems using SAP HANA",
    "tags": ['SAP HANA', 'SAP ERP', 'XSJS', 'SAPUI5'],
    "url": "",
    "section": "Big Data",
    "category": "projects"
}, {
    "img_url": "img/idt.jpg",
    "title": "DIGITAL TRANSFORMATION LANDSCAPE ANALYSIS",
    "description": "A holistic analysis of simultaneous digital transformation of organizations",
    "tags": ['NetworkX'],
    "url": "",
    "section": "Business",
    "category": "projects"
}, {
    "img_url": "img/1c-sales-prediction.png",
    "title": "1C COMPANY SALES PREDICTION",
    "description": "Predict total sales for every product and store in the next month",
    "tags": ['LGBM', 'CatBoost', 'Vowpal Wabbit', 'sklearn', 'fastai', 'stacking'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/1c-sales-prediction",
    "section": "Business",
    "category": "projects"
}, {
    "img_url": "img/rossmann-sales-prediction.png",
    "title": "ROSSMANN SALES PREDICTION",
    "description": "Forecast daily Rossmann sales for up to six weeks in advance",
    "tags": ['fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/rossmann-sales-prediction",
    "section": "Business",
    "category": "projects"
}, {
    "img_url": "img/credit-card-fraud-detection.png",
    "title": "CREDIT CARD FRAUD DETECTION",
    "description": "Recognize fraudulent credit card transactions with anomaly detection",
    "tags": ['sklearn', 'PyTorch', 'fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/credit-card-fraud-detection",
    "section": "Business",
    "category": "projects"
}, {
    "img_url": "img/movielens-recommendation.jpg",
    "title": "MOVIE RECOMMENDER SYSTEM",
    "description": "Build a movie recommender system using item-based collaborative filtering",
    "tags": ['fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/movielens-recommendation",
    "section": "Business",
    "category": "projects"
}, {
    "img_url": "img/gtd.png",
    "title": "GLOBAL TERRORISM DATABASE",
    "description": "Perform analysis on the most comprehensive open-source database on terrorist events",
    "tags": ['sklearn', 'D3.js'],
    "url": "https://polakowo.github.io/gtd-analysis/project/",
    "section": "Social Data",
    "category": "projects"
}, {
    "img_url": "img/imdb.jpg",
    "title": "IMDB DATABASE",
    "description": "Perform network and sentiment analysis on the IMDB database",
    "tags": ['sklearn', 'NetworkX', 'NLTK'],
    "url": "https://polakowo.github.io/oscarobber/",
    "section": "Social Data",
    "category": "projects"
}, {
    "img_url": "img/cryptoz.jpg",
    "title": "CRYPTOCURRENCY MARKET TRACKING",
    "description": "Track cryptocurrency markets with advanced visualization techniques",
    "tags": [],
    "url": "https://github.com/polakowo/cryptoz",
    "section": "Trading",
    "category": "projects"
}, {
    "img_url": "img/vector-bt.jpg",
    "title": "BACKTESTING AND TRADE OPTIMIZATION",
    "description": "Apply a trading system to historical data to find the best strategy",
    "tags": [],
    "url": "https://github.com/polakowo/vector-bt",
    "section": "Trading",
    "category": "projects"
}, {
    "img_url": "img/airbus-ship-segmentation.png",
    "title": "AIRBUS SHIP SEGMENTATION",
    "description": "Build a model that detects ships in satellite images",
    "tags": ['fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/airbus-ship-segmentation",
    "section": "Computer Vision",
    "category": "projects"
}, {
    "img_url": "img/planet-amazon-classification.jpg",
    "title": "AMAZON FROM SPACE CLASSIFICATION",
    "description": "Label satellite image chips with atmospheric conditions and various classes of land cover/land use",
    "tags": ['fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/planet-amazon-classification",
    "section": "Computer Vision",
    "category": "projects"
}, {
    "img_url": "img/amazon-reviews-sentiment-analysis.jpg",
    "title": "AMAZON REVIEWS SENTIMENT ANALYSIS",
    "description": "Analyze Amazon reviews to predict the sentiments that the reviews express",
    "tags": ['fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/amazon-reviews-sentiment-analysis",
    "section": "Natural Language Processing",
    "category": "projects"
}, {
    "img_url": "img/optimized-transfer-learning.jpg",
    "title": "OPTIMIZED TRANSFER LEARNING",
    "description": "Use an optimized approach to fine-tune Keras models faster",
    "tags": ['Keras'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/optimized-transfer-learning",
    "section": "Miscellaneous",
    "category": "projects"
}, {
    "img_url": "img/visual-model-explanation.png",
    "title": "VISUAL MODEL EXPLANATION",
    "description": "Explain the workings of convolutional networks by using automated interpretation methods",
    "tags": ['PyTorch', 'fastai'],
    "url": "https://github.com/polakowo/machine-learning/tree/master/projects/visual-model-explanation",
    "section": "Miscellaneous",
    "category": "projects"
}]

const generate_section = (name) => $(`
    <div class="row">
        <div class="col">
            <hr>
        </div>
        <div class="col-auto">${name}</div>
        <div class="col">
            <hr>
        </div>
    </div>
`)
const generate_row = () => $('<div class="row"></div>')
const generate_card = (item) => $(`
    <div class="col-md-4">
        <div class="card mb-4 box-shadow">
            <div class="img-container">
                <img class="card-img-top" src="${item.img_url}">
                <div class="text-overlay">
                    <span>${item.title}</span>
                </div>
            </div>
            <div class="card-body">
                <p class="card-text">${item.description}</p>
                ${item.tags.length > 0 ? `<p class="card-text">${item.tags.map(
                    tag => `<span class="badge badge-info">${tag}</span>`).join(' ')}</p>` : ''}
                <div class="d-flex justify-content-between align-items-center">
                    <div class="btn-group">
                        ${item.url ? `<a href="${item.url}"
                            target="_blank" role="button"
                            class="btn btn-sm btn-outline-secondary">View</a>` : `<a href="#"
                            target="_blank" role="button"
                            class="btn btn-sm btn-secondary disabled">Private</a>`}
                    </div>
                </div>
            </div>
        </div>
    </div>
`)

sections = []
cards.forEach((item, i) => {
    section_id = `${item.category}-${item.section}`
    $container = $(`#${item.category}-container`)
    if (!sections.includes(section_id)) {
        sections.push(section_id)
        $container.append(generate_section(item.section))
        $container.append(generate_row())
    }
    $container.find('div.row:last').append(generate_card(item))
})
