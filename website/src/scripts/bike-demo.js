export function initBikeDemo(root, { feasibleValues, illustrativePayloads, permitsPayload, canCommit }) {
  if (!root) return;
  const find = selector => root.querySelector(selector);
  const stages = [...root.querySelectorAll('[data-stage]')];
  const tokens = [...root.querySelectorAll('[data-token-index]')];
  let state = { total: 20, casual: 15, field: 'casual', prefix: '', complete: false };

  function render() {
    const values = feasibleValues(state.total, state.casual, state.field);
    const isCasual = state.field === 'casual';
    find('#bike-total').value = state.total;
    find('[data-total-output]').textContent = state.total;
    find('#bike-casual').max = state.total;
    find('#bike-casual').value = state.casual;
    find('#bike-casual').disabled = isCasual;
    find('[data-casual-output]').textContent = isCasual ? 'Not committed' : state.casual;
    find('[data-casual-hint]').textContent = isCasual ? 'Choose a token and commit casual to constrain registered.' : 'Changing casual updates the remaining registered count.';
    stages.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.stage === state.field)));
    find('[data-row-total]').textContent = state.total;
    find('[data-row-casual]').textContent = isCasual ? (state.prefix || '?') : state.casual;
    find('[data-row-registered]').textContent = isCasual ? '?' : (state.prefix || '?');
    find('[data-row-casual]').classList.toggle('current-field', isCasual);
    find('[data-row-registered]').classList.toggle('current-field', !isCasual);
    root.classList.toggle('complete', state.complete);
    const set = values.length === 1 ? values[0] : `0, …, ${state.total}`;
    find('[data-support-field]').textContent = state.field;
    find('[data-support-values]').textContent = `{ ${set} }`;
    find('[data-support-explanation]').textContent = isCasual ? `Any casual count from 0 to ${state.total} leaves a nonnegative registered count.` : `With total = ${state.total} and casual = ${state.casual}, registered must equal ${state.total - state.casual}.`;
    find('[data-prefix-field]').textContent = `${state.field} is`;
    find('#bike-prefix').value = state.prefix;
    const payloads = illustrativePayloads(values, state.prefix);
    tokens.forEach((button, index) => {
      const payload = payloads[index];
      const allowed = permitsPayload(values, state.prefix, payload);
      button.dataset.payload = payload;
      button.disabled = !allowed || state.complete;
      button.classList.toggle('masked', !allowed);
      button.setAttribute('aria-label', `Token ${payload}: ${allowed ? 'has a legal completion' : 'masked, no legal completion'}`);
      button.querySelector('.token-payload').textContent = payload;
      button.querySelector('.token-state').textContent = allowed ? '✓' : '×';
    });
    const committable = canCommit(values, state.prefix);
    find('[data-commit]').disabled = !committable || state.complete;
    find('[data-status]').textContent = state.complete ? `Complete: ${state.total} = ${state.casual} + ${state.total - state.casual}.` : !state.prefix ? 'Choose a token to begin.' : committable ? `${state.prefix} is a complete legal value. You may commit it.` : values.some(value => String(value).startsWith(state.prefix)) ? 'Feasible prefix. More digits are needed before committing.' : 'This prefix has no legal completion. Clear it to continue.';
  }

  function clearPrefix() { state.prefix = ''; state.complete = false; render(); }
  function commit() {
    if (state.complete || !canCommit(feasibleValues(state.total, state.casual, state.field), state.prefix)) return;
    if (state.field === 'casual') { state.casual = Number(state.prefix); state.field = 'registered'; state.prefix = ''; }
    else state.complete = true;
    render();
  }

  find('#bike-total').addEventListener('input', event => { state.total = Number(event.target.value); state.casual = Math.min(state.casual, state.total); clearPrefix(); });
  find('#bike-casual').addEventListener('input', event => { state.casual = Number(event.target.value); clearPrefix(); });
  stages.forEach(button => button.addEventListener('click', () => { state.field = button.dataset.stage; clearPrefix(); }));
  tokens.forEach(button => button.addEventListener('click', () => {
    if (button.disabled) return;
    state.prefix += button.dataset.payload;
    state.complete = false;
    render();
    find('#bike-prefix').focus({ preventScroll: true });
  }));
  find('#bike-prefix').addEventListener('input', event => { state.prefix = event.target.value; state.complete = false; render(); });
  find('#bike-prefix').addEventListener('keydown', event => { if (event.key === 'Enter') { event.preventDefault(); commit(); } });
  find('[data-clear]').addEventListener('click', () => { clearPrefix(); find('#bike-prefix').focus({ preventScroll: true }); });
  find('[data-reset]').addEventListener('click', () => { state = { total: 20, casual: 15, field: 'casual', prefix: '', complete: false }; render(); });
  find('[data-commit]').addEventListener('click', commit);
  render();
}
