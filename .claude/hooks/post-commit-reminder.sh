#!/usr/bin/env bash
node -e "
let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{
  const cmd=(JSON.parse(d).tool_input||{}).command||'';
  if(/(?:^|\s)git\s+commit(?:\s|$)/.test(cmd))
    console.log('[POST-COMMIT REMINDER] Run /complete-langchain-task to update ai-specs docs.');
});"
