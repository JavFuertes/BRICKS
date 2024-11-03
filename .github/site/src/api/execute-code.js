// src/pages/api/execute-code.js
import { exec } from 'child_process';

export async function post({ request }) {
  const { code } = await request.json();

  // Execute the Python code (ensure sandboxing for safety!)
  return new Promise((resolve, reject) => {
    exec(`python -c "${code}"`, (error, stdout, stderr) => {
      if (error) {
        resolve(new Response(JSON.stringify({ output: stderr }), { status: 500 }));
      } else {
        resolve(new Response(JSON.stringify({ output: stdout }), { status: 200 }));
      }
    });
  });
}