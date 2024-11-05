import { exec } from 'child_process';
export async function post({ request }) {
  const { code } = await request.json();

  if (!isValidCode(code)) {
    return new Response(JSON.stringify({ error: "Invalid code" }), { status: 400 });
  }
  const sanitizedCode = sanitizeCode(code);
  const command = `python -c "import json; result = ${sanitizedCode}; print(json.dumps(result))"`;

  return new Promise((resolve, reject) => {
    exec(command, { timeout: 5000, maxBuffer: 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        console.error("Execution error:", stderr);
        resolve(new Response(JSON.stringify({ output: stderr }), { status: 500 }));
      } else {
        try {
          const parsedOutput = JSON.parse(stdout);
          if (isValidOutput(parsedOutput)) {
            resolve(new Response(JSON.stringify({ output: parsedOutput }), { status: 200 }));
          } else {
            resolve(new Response(JSON.stringify({ error: "Invalid output format" }), { status: 400 }));
          }
        } catch (parseError) {
          console.error("Parsing error:", parseError);
          resolve(new Response(JSON.stringify({ error: "Failed to parse output" }), { status: 500 }));
        }
      }
    });
  });
}

function isValidCode(code) {
  
  if (typeof code !== 'string' || code.trim().length === 0) {
    return false;
  }

  const disallowedPatterns = [/import\s+os/, /import\s+sys/, /exec\(/, /eval\(/];
  for (const pattern of disallowedPatterns) {
    if (pattern.test(code)) {
      return false;
    }
  }
  return true;
}

function sanitizeCode(code) {
  return code.replace(/"/g, '\\"');
}

function isValidOutput(output) {
  
  if (typeof output !== 'object' || Array.isArray(output) || output === null) {
    return false;
  }

  for (const key in output) {
    if (!Array.isArray(output[key])) {
      return false;
    }
    for (const item of output[key]) {
      if (typeof item !== 'string' && typeof item !== 'number') {
        return false;
      }
    }
  }

  return true;
}