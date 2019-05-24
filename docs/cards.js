$.getJSON("cards.json", function (cards) {
    const generate_section = (name) => $(`
    <div class="row">
        <div class="col">
            <hr>
        </div>
        <div class="col-auto">${name}</div>
        <div class="col">
            <hr>
        </div>
    </div>`)
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
    </div>`)

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
})