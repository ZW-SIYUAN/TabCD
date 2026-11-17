import { test } from 'node:test';
import assert from 'node:assert/strict';
import { feasibleValues, permitsPayload, canCommit, illustrativePayloads, DOMAIN_MAX } from '../src/lib/bike-demo.js';

test('supports equal independent enumeration of all legal rows on the illustrated domain', () => {
  for (let total = 0; total <= DOMAIN_MAX; total++) {
    const legalRows = [];
    for (let casual = 0; casual <= DOMAIN_MAX; casual++) {
      for (let registered = 0; registered <= DOMAIN_MAX; registered++) {
        if (casual + registered === total) legalRows.push({ casual, registered });
      }
    }
    assert.deepEqual(feasibleValues(total, 0, 'casual'), legalRows.map(row => row.casual));
    for (let casual = 0; casual <= DOMAIN_MAX; casual++) {
      assert.deepEqual(feasibleValues(total, casual, 'registered'), legalRows.filter(row => row.casual === casual).map(row => row.registered));
    }
  }
});

test('payload checks retain exactly the candidates with a canonical legal completion', () => {
  const prefixes = ['', '0', '1', '2', '5', '6', '00', '15', '25', '61', '-'];
  const payloads = ['0', '1', '2', '5', '15', '25', '60', '61', 'x'];
  for (let total = 0; total <= DOMAIN_MAX; total++) {
    for (const values of [feasibleValues(total, 0, 'casual'), feasibleValues(total, Math.min(15, total), 'registered')]) {
      const spellings = values.map(String);
      for (const prefix of prefixes) {
        assert.equal(canCommit(values, prefix), spellings.includes(prefix));
        for (const payload of payloads) {
          const candidate = prefix + payload;
          assert.equal(permitsPayload(values, prefix, payload), spellings.some(spelling => spelling.slice(0, candidate.length) === candidate));
        }
      }
    }
  }
});

test('the two manuscript steps distinguish a feasible prefix from a committed value', () => {
  const first = feasibleValues(20, 15, 'casual');
  assert.equal(permitsPayload(first, '', '15'), true);
  assert.equal(permitsPayload(first, '', '25'), false);
  assert.equal(permitsPayload(first, '', '1'), true);
  assert.equal(canCommit([15], '1'), false);
  assert.deepEqual(feasibleValues(20, 15, 'registered'), [5]);
  assert.equal(canCommit([5], '05'), false);
  assert.deepEqual(feasibleValues(20, 21, 'registered'), []);
  assert.deepEqual(feasibleValues(-1, 0), []);
});

test('the illustrative candidate subset always offers a continuation for an unfinished legal value', () => {
  for (let value = 0; value <= DOMAIN_MAX; value++) {
    const spelling = String(value);
    for (let length = 0; length < spelling.length; length++) {
      const prefix = spelling.slice(0, length);
      const candidates = illustrativePayloads([value], prefix);
      assert.equal(candidates.length, new Set(candidates).size);
      assert.ok(candidates.some(payload => permitsPayload([value], prefix, payload)));
    }
  }
});
