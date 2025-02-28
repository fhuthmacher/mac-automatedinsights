import { readFileSync } from 'node:fs';
import { execSync } from 'node:child_process';

try {
  const outputs = JSON.parse(readFileSync('src/cdk-outputs.json', 'utf8'));
  const bucket = outputs.GenAiStack.genaiaaWebsiteBucketName;
  
  // this script replaces: // "deploy": "npm run build && node -e \"const fs = require('fs'); const { execSync } = require('child_process'); const outputs = JSON.parse(fs.readFileSync('src/cdk-outputs.json', 'utf8')); const bucket = outputs.ChatappStack.WebsiteBucketName; execSync('aws s3 sync dist s3://' + bucket );console.log('Website deployed to: https://' + outputs.ChatappStack.CloudFrontDistributionDomainName);\""
  // Use --only-show-errors to reduce output buffer size
  // Use --size-only to reduce comparison overhead
  execSync(`aws s3 sync dist s3://${bucket} --only-show-errors --size-only`, {
    stdio: 'inherit', // This will stream the output instead of buffering it
    maxBuffer: 1024 * 1024 * 10 // Increase max buffer to 10MB if needed
  });

  console.log('Website deployed to:', `https://${outputs.GenAiStack.genaiaaCloudFrontDistributionDomainName}`);
} catch (error) {
  console.error('Deployment failed:', error);
  process.exit(1);
}