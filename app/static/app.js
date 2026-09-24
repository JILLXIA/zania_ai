const form = document.querySelector('#upload-form');
const button = document.querySelector('#analyze');
const progress = document.querySelector('#progress');
const error = document.querySelector('#error');
const output = document.querySelector('#output');
let lastResponse = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  error.hidden = output.hidden = true;
  lastResponse = null;
  progress.textContent = 'Reading your document and preparing answers…';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 250000);
  try {
    const response = await fetch('/qa', {method: 'POST', body: new FormData(form), signal: controller.signal});
    const data = await response.json();
    if (!response.ok && !data.results) {
      throw new Error(data.error?.message || `Request failed (${response.status}).`);
    }
    if (data.error) {
      error.textContent = data.error.message;
      error.hidden = false;
    }
    lastResponse = data;
    const results = document.querySelector('#results');
    const warnings = document.querySelector('#warnings');
    results.replaceChildren();
    warnings.replaceChildren();
    for (const warning of data.warnings || []) warnings.append(element('p', warning, 'warning'));
    for (const result of data.results) {
      const card = element('article', undefined, 'answer-card');
      card.append(element('span', result.status.replaceAll('_', ' '), `status ${result.status}`));
      card.append(element('h3', result.question), element('p', result.answer));
      if (result.citations.length) {
        const details = element('details');
        details.append(element('summary', `View evidence (${result.citations.length})`));
        for (const citation of result.citations) {
          const quote = element('blockquote');
          const location = citation.page ? `Page ${citation.page}` : `JSON ${citation.source_path || '(root)'}`;
          quote.append(element('small', `${location} · ${citation.source_type === 'image' ? 'Image analysis' : 'Source text'}`));
          quote.append(element('span', citation.excerpt));
          details.append(quote);
        }
        card.append(details);
      }
      results.append(card);
    }
    output.hidden = false;
    progress.textContent = `${data.results.length} questions processed.`;
  } catch (exc) {
    error.textContent = exc.name === 'AbortError' ? 'The request timed out. Try a smaller document.' : exc.message;
    error.hidden = false;
    progress.textContent = '';
  } finally {
    clearTimeout(timeout);
    button.disabled = false;
  }
});

document.querySelector('#download').addEventListener('click', () => {
  if (!lastResponse) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResponse, null, 2)], {type: 'application/json'}));
  const link = element('a');
  link.href = url;
  link.download = 'answers.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
