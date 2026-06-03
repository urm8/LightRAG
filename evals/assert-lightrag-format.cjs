const { spawnSync } = require('node:child_process');

module.exports = function assertLightRagFormat(output, context) {
  const python = process.env.PYTHON || '.venv/bin/python';
  const check = spawnSync(python, ['evals/check_lightrag_parser.py'], {
    input: JSON.stringify({
      output,
      vars: context?.vars || {},
    }),
    encoding: 'utf8',
  });

  if (check.error) {
    return {
      pass: false,
      score: 0,
      reason: `Parser check failed to start: ${check.error.message}`,
    };
  }

  if (check.status !== 0) {
    return {
      pass: false,
      score: 0,
      reason: `Parser check exited ${check.status}: ${check.stderr || check.stdout}`,
    };
  }

  let result;
  try {
    result = JSON.parse(check.stdout);
  } catch (error) {
    return {
      pass: false,
      score: 0,
      reason: `Parser check returned invalid JSON: ${error.message}\n${check.stdout}`,
    };
  }

  const errors = [];
  if (!result.token_budget_ok) {
    errors.push(
      `input_token_budget_exceeded: ${result.input_tokens}/${result.input_budget}`
    );
  }
  for (const warningClass of result.warning_classes || []) {
    errors.push(`parser_warning:${warningClass}`);
  }
  for (const manualError of result.manual_errors || []) {
    errors.push(manualError);
  }

  if (errors.length) {
    return {
      pass: false,
      score: 0,
      reason: errors.slice(0, 40).join('\n'),
    };
  }

  return {
    pass: true,
    score: 1,
    reason: `LightRAG parser accepted ${result.node_count} entities and ${result.edge_count} relations within ${result.input_tokens}/${result.input_budget} input tokens`,
  };
};
