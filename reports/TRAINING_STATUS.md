# Training Status Dashboard

**Started**: $(date)
**Budget per platform**: $3,000

## Training Jobs

| Platform | Agent ID | Status | Started | Cost Est. |
|----------|----------|--------|---------|-----------|
| TPU v5e-8 | a291200 | 🔄 Running | $(date +%H:%M) | ~$2.04/hr |
| TPU v6e-4 | afd821a | 🔄 Running | $(date +%H:%M) | ~$1.70/hr |
| Vertex AI | a7727c2 | 🔄 Running | $(date +%H:%M) | TBD |

## Monitoring Schedule
- Check interval: 30 minutes
- Next check: $(date -d '+30 minutes' +%H:%M)

## Pipeline Stages
1. ⬜ Base Training (pretraining)
2. ⬜ SFT Training (supervised fine-tuning)
3. ⬜ GSPO Training (group sequence policy optimization)
4. ⬜ Evaluation (HumanEval C++)
5. ⬜ Report Generation

## Cost Tracking
- GCS Storage: ~$0.02/GB/month (~$0.60/month for 30GB)
- TPU v5e-8 spot: $2.04/hr
- TPU v6e-4 spot: $1.70/hr
- Vertex AI: varies by machine type

---
*Last updated: $(date)*
