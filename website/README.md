# TabCD project website

The paper project page and interactive Bike demo are published at
**https://zw-siyuan.github.io/TabCD/**. Website source lives in `website/`; the
Python implementation in the repository root is maintained separately.

The page uses [Roman Hauksson's Academic Project Page Template](https://github.com/RomanHauksson/academic-project-astro-template),
based on Astro and MDX. Template commit: `0e90d4f36bad6ee0af330964bdf0002fe451283d`.
The build uses security-patched Astro 7 and the unified Markdown pipeline while
preserving the template's layout and whitespace behavior.

## Edit and preview

Use Node.js 24. From the repository root:

```sh
cd website
npm ci
npm run dev
```

Open the local URL printed by Astro. The main page content is `src/paper.mdx`.
The Code button links to this repository; the Paper button downloads
`public/ICDE2027.pdf`.

To check the demo and build a production version:

```sh
npm test
npm run build
node scripts/serve.mjs
```

The static preview serves both `/` and `/TabCD/`. To reproduce the Pages build
locally, set `SITE_URL=https://zw-siyuan.github.io` and `BASE_PATH=/TabCD` before
running the build. These variables are supplied automatically by the workflow.

## Publish updates

The root workflow, [`.github/workflows/pages.yml`](../.github/workflows/pages.yml),
runs when a commit on `main` changes `website/` or the workflow itself. It installs
the locked npm dependencies, runs the demo's semantic tests and Astro checks,
builds the site, and deploys only `website/dist/` to GitHub Pages. Changes to the
Python implementation do not trigger a website rebuild.

In the repository's **Settings → Pages**, the publishing source is
**GitHub Actions**. A manual rebuild is also available through
**Actions → Publish project website → Run workflow**.

Only source and the paper assets are tracked. Do not commit `node_modules/`,
`dist/`, `.astro/`, credentials, or local verification files.

## Content and assets

- `src/paper.mdx`: title, authors, abstract, method, demo placement, and results.
- `src/components/BikeDemo.astro`: interactive demonstration.
- `src/lib/bike-demo.js` and `src/scripts/bike-demo.js`: finite-domain semantics
  and UI behavior. The demo illustrates `total = casual + registered` over
  integers 0–60 and does not perform model inference.
- `public/ICDE2027.pdf`: the supplied manuscript, copied without modification.
- `public/citation.bib`: manuscript citation for the four authors.
- `src/assets/paper/`: four figures rendered directly from the supplied PDF.
- `PAPER_PROVENANCE.json`: PDF hash, figure numbers, crop coordinates, and hashes.
- `SOURCE_PROVENANCE.json`: template and source attribution.

If the manuscript changes, review its authors, figures, and claims before
updating the page. The optional figure preparation helper requires Python and
PyMuPDF and is not part of deployment:

```sh
python scripts/prepare-paper-assets.py /path/to/ICDE2027.pdf
```

Update its page/crop coordinates for a revised paper before running it.

## Attribution and licenses

The website template and its adaptations use **CC BY-SA 4.0**; the page footer
retains attribution and the license link. See [TEMPLATE_NOTICE.md](TEMPLATE_NOTICE.md).
The paper and its figures remain the authors' work. The Python code's MIT
license is recorded separately in the repository root.
