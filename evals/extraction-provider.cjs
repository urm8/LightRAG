/**
 * extraction-provider.cjs
 *
 * Promptfoo provider for LightRAG entity extraction evaluation.
 * Reads the active prompts from evals/prompts.active.json and calls the
 * configured extraction LLM (an OpenAI-compatible endpoint).
 *
 * Env vars (all optional, with fallbacks):
 *   EXTRACTION_LLM_BINDING_HOST — base URL (default: http://127.0.0.1:11438/v1)
 *   EXTRACTION_LLM_BINDING_API_KEY — API key (default: dummy)
 *   EXTRACTION_LLM_MODEL — model name (default: huihui-ai/Huihui-granite-4.1-3b-abliterated)
 *   EXTRACTION_LLM_MAX_TOKENS — max completion tokens (default: 512)
 */

const fs = require('fs');

const prompts = JSON.parse(
  fs.readFileSync('evals/prompts.active.json', 'utf8')
);

function fill(template, vars) {
  return template
    .replaceAll('{entity_types}', vars.entity_types || 'Person, Organization, Location, Event, Concept, Method, Content, Data, Artifact, Workspace, Project, Repository, Directory, File, ProgrammingLanguage, TechnologyStack, Framework, Library, Runtime, Service, Deployment, Environment, Configuration, Command, APIEndpoint, Database, StorageBackend, Other')
    .replaceAll('{language}', vars.language || 'English')
    .replaceAll('{tuple_delimiter}', prompts.tuple_delimiter)
    .replaceAll('{completion_delimiter}', prompts.completion_delimiter)
    .replaceAll('{input_text}', vars.input_text || '');
}

class ExtractionProvider {
  id() {
    return 'lightrag-extraction';
  }

  async callApi(prompt, context) {
    const vars = context.vars || {};

    const baseUrl = process.env.EXTRACTION_LLM_BINDING_HOST
      || process.env.APFEL_OPENAI_BASE_URL
      || 'http://127.0.0.1:11438/v1';
    const apiKey = process.env.EXTRACTION_LLM_BINDING_API_KEY
      || process.env.APFEL_OPENAI_API_KEY
      || 'dummy';
    const model = process.env.EXTRACTION_LLM_MODEL
      || process.env.APFEL_MODEL
      || 'huihui-ai/Huihui-granite-4.1-3b-abliterated';
    const maxTokens = Number(
      process.env.EXTRACTION_LLM_MAX_TOKENS
      || process.env.APFEL_MAX_TOKENS
      || '512'
    );

    const system = fill(prompts.system, vars);
    const user = fill(prompts.user, vars);

    const maxRetries = 2;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
          temperature: 0,
          max_tokens: maxTokens,
        }),
      });

      if (res.ok) {
        const json = await res.json();
        const output = json.choices?.[0]?.message?.content || '';
        return { output };
      }

      if (res.status === 400 && attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        continue;
      }

      const text = await res.text();
      return {
        error: `Extraction LLM HTTP ${res.status}: ${text}`,
      };
    }
  }
}

module.exports = ExtractionProvider;
