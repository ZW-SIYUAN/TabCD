// An educational finite-domain example, independent of the research runtime.
// All values are canonical nonnegative integers on the displayed 0..60 domain.
export const DOMAIN_MAX = 60;

export function feasibleValues(total, casual, field = 'casual') {
  if (!Number.isInteger(total) || total < 0 || total > DOMAIN_MAX) return [];
  if (field === 'casual') return Array.from({ length: total + 1 }, (_, i) => i);
  if (field !== 'registered' || !Number.isInteger(casual) || casual < 0 || casual > total) return [];
  return [total - casual];
}

export function permitsPayload(values, prefix, payload) {
  if (typeof prefix !== 'string' || typeof payload !== 'string' || !/^\d*$/.test(prefix + payload)) return false;
  return values.some(value => String(value).startsWith(prefix + payload));
}

export function canCommit(values, prefix) {
  return prefix !== '' && values.some(value => String(value) === prefix);
}

export function illustrativePayloads(values, prefix) {
  const candidates = ['1', '15', '25', '5', '0', '2'];
  if (!candidates.some(payload => permitsPayload(values, prefix, payload)) && !canCommit(values, prefix)) {
    const spelling = values.map(String).find(value => value.startsWith(prefix) && value.length > prefix.length);
    if (spelling !== undefined) candidates[candidates.length - 1] = spelling[prefix.length];
  }
  return candidates;
}
