# Strix Testing Program: Resource Estimation

## Executive Summary

**Program Duration:** 6 months (Initial Development) + Ongoing Maintenance  
**Weekly Build Cadence:** 52 builds/year  
**Total Test Suites:** 21 suites across 8 categories  
**Recommended Team Size:** 8-10 FTE (Full-Time Equivalents)

---

## Resource Breakdown

### Phase 1: Initial Development (Months 1-3)

#### Team Composition

| Role | FTE | Responsibilities |
|------|-----|------------------|
| **Test Development Lead** | 1.0 | Architecture, test framework design, code reviews |
| **Senior Test Engineers** | 2.0 | Test implementation, complex model testing (7B models) |
| **Test Engineers** | 2.0 | Test implementation, small/medium models (<5B) |
| **DevOps Engineer** | 1.0 | CI/CD integration, infrastructure, automation |
| **Performance Engineer** | 1.0 | Profiling, benchmarking, optimization analysis |
| **QA/Validation Engineer** | 1.0 | Test execution, validation, bug triage |
| **Technical Writer** | 0.5 | Documentation, test guides, runbooks |

**Total Phase 1:** 8.5 FTE

---

### Phase 2: Stabilization & Optimization (Months 4-6)

#### Team Composition

| Role | FTE | Responsibilities |
|------|-----|------------------|
| **Test Development Lead** | 0.5 | Code reviews, escalations, optimization |
| **Senior Test Engineers** | 1.5 | Bug fixes, optimization, edge case handling |
| **Test Engineers** | 1.0 | Test refinement, additional coverage |
| **DevOps Engineer** | 1.0 | CI/CD hardening, scalability, monitoring |
| **Performance Engineer** | 1.0 | Performance tuning, bottleneck analysis |
| **QA/Validation Engineer** | 1.0 | Full regression testing, acceptance testing |
| **Technical Writer** | 0.5 | Final documentation, training materials |

**Total Phase 2:** 6.5 FTE

---

### Phase 3: Steady State Maintenance (Month 7+)

#### Team Composition

| Role | FTE | Responsibilities |
|------|-----|------------------|
| **Test Development Lead** | 0.25 | Oversight, strategic planning, quarterly reviews |
| **Senior Test Engineer** | 1.0 | New model additions, complex debugging, mentoring |
| **Test Engineer** | 1.0 | Test maintenance, updates, bug fixes |
| **DevOps Engineer** | 0.5 | CI/CD maintenance, infrastructure updates |
| **Performance Engineer** | 0.5 | Weekly performance analysis, trend monitoring |
| **QA/Validation Engineer** | 0.5 | Weekly validation, release certification |

**Total Steady State:** 3.75 FTE

---

## Effort Estimation by Test Category

### Development Effort (Person-Weeks)

| Category | # Tests | Complexity | Dev Effort | Test/Debug | Total PW |
|----------|---------|------------|------------|------------|----------|
| **VLM** | 4 | Medium | 8 PW | 4 PW | 12 PW |
| **VLA (Robotics)** | 3 | High | 9 PW | 6 PW | 15 PW |
| **Multimodal** | 2 | High | 6 PW | 4 PW | 10 PW |
| **Segmentation** | 1 | Medium | 2 PW | 1 PW | 3 PW |
| **LLM** | 1 | Medium | 2 PW | 1 PW | 3 PW |
| **ASR** | 3 | Medium | 6 PW | 3 PW | 9 PW |
| **Quantization** | 3 | High | 9 PW | 6 PW | 15 PW |
| **Strix-Specific** | 4 | High | 12 PW | 8 PW | 20 PW |
| **CI/CD Integration** | - | High | 8 PW | 4 PW | 12 PW |
| **Infrastructure** | - | Medium | 4 PW | 2 PW | 6 PW |
| **Documentation** | - | Low | 4 PW | 1 PW | 5 PW |

**Total Development Effort:** 110 Person-Weeks (~22 months of effort)

**With 8.5 FTE:** 110 PW / 8.5 FTE = **13 weeks (3 months)**

---

## Weekly Build Activities

### Per-Build Effort (1 build/week)

| Activity | Role | Time/Week | Automation Level |
|----------|------|-----------|------------------|
| **Pre-Build Validation** | DevOps | 2h | 80% automated |
| **Build Trigger & Monitoring** | DevOps | 1h | 90% automated |
| **Test Execution** | Automated | 4-8h | 100% automated |
| **Result Analysis** | QA | 3h | 50% automated |
| **Performance Review** | Perf Engineer | 2h | 70% automated |
| **Bug Triage** | Test Lead | 1h | Manual |
| **Regression Investigation** | Test Engineer | 2-4h | As needed |
| **Report Generation** | QA | 1h | 80% automated |
| **Stakeholder Review** | Test Lead | 1h | Manual |

**Total Weekly Effort:** 13-18 hours (~0.35-0.45 FTE per week)

**With 52 builds/year:** ~18-23 FTE-weeks/year

---

## Cost Analysis (Annual, Steady State)

### Assumptions
- Blended rate: $150K/year per FTE (includes benefits, overhead)
- Infrastructure costs: $50K/year (hardware, cloud, licenses)
- Training & misc: $20K/year

### Steady State Cost Breakdown

| Item | FTE | Cost/Year |
|------|-----|-----------|
| **Test Development Lead** | 0.25 | $37,500 |
| **Senior Test Engineer** | 1.0 | $150,000 |
| **Test Engineer** | 1.0 | $150,000 |
| **DevOps Engineer** | 0.5 | $75,000 |
| **Performance Engineer** | 0.5 | $75,000 |
| **QA/Validation Engineer** | 0.5 | $75,000 |
| **Infrastructure** | - | $50,000 |
| **Training & Misc** | - | $20,000 |

**Total Annual Cost (Steady State):** $632,500

**Per Build Cost:** $632,500 / 52 = ~$12,163/build

---

## Initial Investment Cost (First 6 Months)

### Phase 1 (Months 1-3)
- 8.5 FTE × 3 months × $12,500/month = $318,750
- Infrastructure setup: $30,000
- Training: $10,000
- **Phase 1 Total:** $358,750

### Phase 2 (Months 4-6)
- 6.5 FTE × 3 months × $12,500/month = $243,750
- Infrastructure: $15,000
- Documentation: $5,000
- **Phase 2 Total:** $263,750

**Total Initial Investment:** $622,500

---

## Resource Allocation Timeline

### Gantt-Style View

```
Month:     1    2    3    4    5    6    7    8    9   10   11   12
───────────────────────────────────────────────────────────────────
Test Lead: ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░
Sr Eng 1:  ███████████████████████████████████████░░░░░░░░░░░░░░░░
Sr Eng 2:  ███████████████████████████████████████░░░░░░░░░░░░░░░░
Eng 1:     ████████████████████████████████████████████████████████
Eng 2:     ████████████████████████████████████████████████████████
DevOps:    ████████████████████████████████████████████████████████
Perf Eng:  ████████████████████████████████████████████████████████
QA:        ████████████████████████████████████████████████████████
Tech Writ: ░░░░░░░░██████████████████████████████████░░░░░░░░░░░░░
───────────────────────────────────────────────────────────────────
           Phase 1: Dev      Phase 2: Stab    Phase 3: Maintenance
           8.5 FTE           6.5 FTE          3.75 FTE

Legend: █ Full-time  ░ Part-time
```

---

## Risk & Contingency

### Recommended Contingency: 20% additional capacity

**Reasons:**
1. **Model Availability:** Some models may require licensing/access delays
2. **Hardware Availability:** Strix hardware may have limited availability
3. **Integration Complexity:** Unforeseen technical challenges
4. **Dependency Changes:** Updates to ROCm, PyTorch, transformers
5. **Staff Turnover:** Knowledge transfer needs

**Adjusted Team Size:** 3.75 FTE × 1.2 = **4.5 FTE (Steady State)**

---

## Scalability Scenarios

### Scenario 1: Add New Model Category (e.g., Diffusion Models)
- **Effort:** 4-6 person-weeks
- **Team:** 1-2 engineers for 2-3 weeks
- **Cost:** $15K-$22K

### Scenario 2: Daily Builds Instead of Weekly
- **Additional FTE:** +1.0 QA/Validation Engineer
- **Additional Cost:** +$150K/year
- **Automation Investment:** $50K (one-time)

### Scenario 3: Add Windows Strix Platform Support
- **Effort:** 20-30 person-weeks
- **Team:** 2 engineers for 3 months
- **Cost:** $75K-$112K

---

## ROI Analysis

### Test Coverage Value

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Bugs Found Pre-Release** | 30-50/year | $500K-$1M saved (customer impact) |
| **Performance Regressions Caught** | 10-15/year | $200K-$400K saved (optimization time) |
| **Release Confidence** | 95%+ | Reduced customer escalations |
| **Time-to-Market** | -20% | Faster iteration cycles |

### Cost-Benefit

**Annual Cost:** $632,500 (steady state)  
**Value Delivered:** $700K-$1.4M (bugs + regressions + time savings)  
**Net Benefit:** $67K-$767K/year  
**ROI:** 11%-121%

---

## Comparison with Industry Benchmarks

### Test Automation Coverage

| Metric | Strix Plan | Industry Avg | Best-in-Class |
|--------|------------|--------------|---------------|
| Automation Level | 85% | 60% | 90% |
| FTE per 100 Tests | 17.8 | 25 | 15 |
| Cost per Test Suite | $30,119 | $40,000 | $25,000 |
| Weekly Build Time | 4-8h | 12h | 3-4h |

**Verdict:** Plan is competitive with industry standards, approaching best-in-class.

---

## Alternative Staffing Models

### Option 1: Fully Outsourced
- **Cost:** $400K-$500K/year
- **Pros:** Lower direct cost, flexible capacity
- **Cons:** Knowledge retention, communication overhead, quality control
- **Recommendation:** ❌ Not recommended for critical Strix testing

### Option 2: Hybrid (2 FTE + Contractors)
- **Core Team:** 2 FTE ($300K)
- **Contractors:** 2-3 contractors as needed ($150K-$200K)
- **Total:** $450K-$500K/year
- **Pros:** Balance of control and flexibility
- **Cons:** Contractor ramp-up time
- **Recommendation:** ⚠️ Viable if budget-constrained

### Option 3: Full In-House (Recommended)
- **Team:** 3.75-4.5 FTE ($632K-$675K)
- **Pros:** Full control, knowledge retention, quality, responsiveness
- **Cons:** Higher fixed cost
- **Recommendation:** ✅ Best for long-term program

---

## Minimum Viable Team (Budget-Constrained)

If budget is limited, absolute minimum:

| Role | FTE | Critical? |
|------|-----|-----------|
| **Senior Test Engineer** | 1.0 | ✅ Must have |
| **Test Engineer** | 1.0 | ✅ Must have |
| **DevOps Engineer (shared)** | 0.25 | ✅ Must have |
| **Performance Engineer (shared)** | 0.25 | ⚠️ Highly recommended |

**Minimum Team:** 2.5 FTE ($375K/year)

**Tradeoffs:**
- Slower feature development
- Limited optimization work
- Higher risk of missing issues
- Dependency on shared resources

---

## Summary Recommendations

### For Weekly Build Cadence with 21 Test Suites:

**Phase 1 (Months 1-3):**
- **Team Size:** 8-9 FTE
- **Investment:** $360K
- **Goal:** Implement all 21 test suites

**Phase 2 (Months 4-6):**
- **Team Size:** 6-7 FTE
- **Investment:** $265K
- **Goal:** Stabilize, optimize, document

**Steady State (Month 7+):**
- **Team Size:** 4-5 FTE
- **Annual Cost:** $630K-$675K
- **Goal:** Maintain, enhance, weekly validation

### Key Success Factors:
1. ✅ Strong test automation (85%+)
2. ✅ Dedicated DevOps for CI/CD reliability
3. ✅ Performance engineer for regression detection
4. ✅ Clear escalation paths and ownership
5. ✅ 20% contingency buffer for unknowns

### Bottom Line:
**Realistic Team Size: 4-5 FTE for steady state operations**  
**Annual Budget: $630K-$675K**  
**Per-Build Cost: ~$12,000-$13,000**  
**ROI: Positive (11%-121% depending on value attribution)**

This staffing level ensures:
- ✅ All 21 test suites maintained
- ✅ Weekly builds validated within 1 business day
- ✅ Performance regressions caught proactively
- ✅ New models added quarterly
- ✅ Documentation kept current
- ✅ Team has bandwidth for innovation

