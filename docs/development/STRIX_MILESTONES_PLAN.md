# Strix Testing Program: Milestone-Based Project Plan

## Program Overview

**Total Duration:** 24 weeks (6 months)  
**Weekly Build Cadence:** Starting Week 8  
**Total Milestones:** 8 major milestones  
**Go/No-Go Gates:** 3 decision points

---

## Milestone Summary

| Milestone | Week | Deliverables | Team Size | Gate |
|-----------|------|--------------|-----------|------|
| **M0: Program Kickoff** | Week 1 | Project charter, team onboarded | 8.5 FTE | ✅ Start |
| **M1: Foundation** | Week 4 | Test framework, CI/CD pipeline | 8.5 FTE | 🚦 Gate 1 |
| **M2: Phase 1 Tests** | Week 8 | Small models (<1B) working | 8.5 FTE | 🚦 Gate 2 |
| **M3: Phase 2 Tests** | Week 12 | Medium models (3-5B) working | 8.5 FTE | - |
| **M4: Phase 3 Tests** | Week 16 | Large models (7B) working | 6.5 FTE | - |
| **M5: Integration** | Week 20 | All 21 tests in CI/CD | 6.5 FTE | 🚦 Gate 3 |
| **M6: Validation** | Week 22 | Performance validated | 6.5 FTE | - |
| **M7: Production** | Week 24 | Weekly builds operational | 4.5 FTE | ✅ Launch |

---

## Detailed Milestone Breakdown

---

## M0: Program Kickoff (Week 1)

### Objectives
- Establish team and infrastructure
- Define success criteria
- Get organizational buy-in

### Deliverables

| # | Deliverable | Owner | Status |
|---|-------------|-------|--------|
| 1 | Project Charter signed | Test Lead | □ |
| 2 | Team hired/assigned | Management | □ |
| 3 | Hardware allocated (Strix devices) | DevOps | □ |
| 4 | ROCm environment installed | DevOps | □ |
| 5 | Access provisioned (GitHub, infra) | DevOps | □ |
| 6 | Weekly standup schedule | Test Lead | □ |
| 7 | Communication plan | Test Lead | □ |

### Team Composition
- Test Development Lead: 1.0 FTE
- Senior Test Engineers: 2.0 FTE
- Test Engineers: 2.0 FTE
- DevOps Engineer: 1.0 FTE
- Performance Engineer: 1.0 FTE
- QA Engineer: 1.0 FTE
- Technical Writer: 0.5 FTE

**Total:** 8.5 FTE

### Success Criteria
- ✅ All team members onboarded
- ✅ Strix hardware accessible
- ✅ ROCm 6.2+ verified functional
- ✅ Kickoff meeting held with stakeholders

### Budget
**Week 1:** $42K

---

## M1: Foundation Complete (Week 4)

### Objectives
- Build test framework foundation
- Establish CI/CD pipeline
- Create first smoke test

### Deliverables

| # | Deliverable | Owner | Status |
|---|-------------|-------|--------|
| 1 | Test framework architecture doc | Test Lead | □ |
| 2 | CI/CD pipeline configured | DevOps | □ |
| 3 | Test fixtures and utilities | Sr Engineer 1 | □ |
| 4 | Strix device detection logic | Sr Engineer 2 | □ |
| 5 | Logging and reporting framework | Engineer 1 | □ |
| 6 | First smoke test (rocblas) | Engineer 2 | □ |
| 7 | Baseline performance metrics | Perf Engineer | □ |
| 8 | Test documentation template | Tech Writer | □ |

### Technical Components

**Test Framework:**
```python
# Core fixtures established
- strix_device fixture
- test_image_224 fixture
- model_loader fixture
- profiling_context fixture
```

**CI/CD Pipeline:**
```yaml
# .github/workflows/strix_tests.yml
- Manual trigger (workflow_dispatch)
- Scheduled runs (weekly)
- Test matrix generation
- Artifact management
```

### Success Criteria
- ✅ Smoke test runs successfully on Strix Linux
- ✅ CI/CD pipeline executes end-to-end
- ✅ Test reports generated automatically
- ✅ Baseline metrics documented

### Budget
**Weeks 1-4:** $168K cumulative

### 🚦 GATE 1: Foundation Review

**Decision Point:** Continue to test development?

**Criteria:**
- [ ] CI/CD pipeline functional
- [ ] At least 1 test passing on Strix hardware
- [ ] Team velocity acceptable (on track)
- [ ] No major technical blockers

**Outcomes:**
- ✅ **GO:** Proceed to M2 (test development)
- ❌ **NO-GO:** Address blockers, replanning required

---

## M2: Phase 1 Tests Complete (Week 8)

### Objectives
- Implement high-priority small model tests
- Achieve first weekly build cycle
- Validate test framework scalability

### Deliverables

| # | Deliverable | Test Category | Owner | Status |
|---|-------------|---------------|-------|--------|
| 1 | Qwen-VL 3B tests | VLM | Sr Engineer 1 | □ |
| 2 | CLIP/BLIP tests (expanded) | VLM | Engineer 1 | □ |
| 3 | SAM2 segmentation tests | Segmentation | Engineer 2 | □ |
| 4 | zipformer ASR tests | ASR | Sr Engineer 2 | □ |
| 5 | Performance profiling setup | Infrastructure | Perf Engineer | □ |
| 6 | Test execution report | Documentation | Tech Writer | □ |

### Test Coverage

| Model | Size | Platforms | Tests | Status |
|-------|------|-----------|-------|--------|
| Qwen2.5-VL-3B | 3B | Linux + Windows | 8 test cases | □ |
| CLIP | 0.5B | Linux + Windows | 5 test cases | □ |
| BLIP | 0.5B | Linux + Windows | 5 test cases | □ |
| SAM2 | 0.2B | Linux + Windows | 6 test cases | □ |
| zipformer | <0.3B | Linux + Windows | 4 test cases | □ |

**Total:** 28 test cases across 5 models

### Success Criteria
- ✅ All 4 test suites passing on Linux Strix
- ✅ At least 3 test suites passing on Windows Strix
- ✅ First weekly build completed successfully
- ✅ Test execution time < 4 hours
- ✅ No critical infrastructure issues

### Budget
**Weeks 1-8:** $336K cumulative

### 🚦 GATE 2: Phase 1 Review

**Decision Point:** Proceed to Phase 2 (medium/large models)?

**Criteria:**
- [ ] All Phase 1 tests operational
- [ ] Weekly build cadence established
- [ ] Performance metrics within targets
- [ ] Team capacity sufficient for Phase 2

**Outcomes:**
- ✅ **GO:** Proceed to M3 (medium models)
- ⚠️ **CONDITIONAL:** Complete Phase 1 fixes, then proceed
- ❌ **NO-GO:** Reassess approach, get stakeholder input

---

## M3: Phase 2 Tests Complete (Week 12)

### Objectives
- Implement medium model tests (3-5B)
- Add quantization validation
- Expand coverage to robotics domain

### Deliverables

| # | Deliverable | Test Category | Owner | Status |
|---|-------------|---------------|-------|--------|
| 1 | Qwen3-Instruct 4B tests | LLM | Engineer 1 | □ |
| 2 | Pi0 robotics tests | VLA | Sr Engineer 2 | □ |
| 3 | crossformer ASR tests | ASR | Engineer 2 | □ |
| 4 | AWQ quantization tests | Quantization | Sr Engineer 1 | □ |
| 5 | Strix memory constraint tests | Strix-Specific | Sr Engineer 1 | □ |
| 6 | Performance comparison report | Documentation | Perf Engineer | □ |

### Test Coverage

| Model | Size | Quantization | Platforms | Tests | Status |
|-------|------|--------------|-----------|-------|--------|
| Qwen3-Instruct | 4B | AWQ | Linux + Windows | 6 test cases | □ |
| Pi0 | 0.5B | - | Linux + Windows | 5 test cases | □ |
| crossformer | <0.3B | - | Linux + Windows | 4 test cases | □ |
| AWQ Validation | Various | AWQ | Linux + Windows | 10 test cases | □ |

**Total:** 25 new test cases (53 cumulative)

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Qwen3-Instruct P50 Latency | <200ms | rocprofv3 |
| Pi0 Inference FPS | >30 | Test harness |
| AWQ Memory Reduction | >70% | vs FP16 baseline |
| Test Suite Runtime | <6 hours | CI/CD logs |

### Success Criteria
- ✅ All Phase 2 tests passing (Linux + Windows)
- ✅ AWQ quantization validated
- ✅ Performance targets met
- ✅ 4 consecutive weekly builds successful

### Budget
**Weeks 1-12:** $504K cumulative

---

## M4: Phase 3 Tests Complete (Week 16)

### Objectives
- Implement large model tests (7B, Linux only)
- Complete all VLA/multimodal tests
- Finalize quantization validation

### Deliverables

| # | Deliverable | Test Category | Owner | Status |
|---|-------------|---------------|-------|--------|
| 1 | Qwen-VL 7B tests (Linux) | VLM | Sr Engineer 1 | □ |
| 2 | Qwen-Omni 7B tests (Linux) | Multimodal | Sr Engineer 2 | □ |
| 3 | OpenVLA tests (Linux) | VLA | Sr Engineer 1 | □ |
| 4 | CogACT tests (Linux) | VLA | Engineer 1 | □ |
| 5 | GPTQ quantization tests | Quantization | Engineer 2 | □ |
| 6 | Qwen3-VL 4B tests | VLM | Engineer 1 | □ |

### Test Coverage

| Model | Size | Platforms | Memory | Tests | Status |
|-------|------|-----------|--------|-------|--------|
| Qwen2.5-VL-7B | 7B | Linux only | ~14GB | 8 test cases | □ |
| Qwen2.5-Omni | 7B | Linux only | ~14GB | 6 test cases | □ |
| Qwen3-Omni | 7B | Linux only | ~14GB | 6 test cases | □ |
| OpenVLA | 7B | Linux only | ~14GB | 8 test cases | □ |
| CogACT | 7B | Linux only | ~14GB | 6 test cases | □ |
| Qwen3-VL | 4B | Linux + Windows | ~8GB | 8 test cases | □ |
| GPTQ Validation | Various | Linux + Windows | - | 10 test cases | □ |

**Total:** 52 new test cases (105 cumulative)

### Team Transition

**Week 16:** Transition to stabilization team

| Role | Before (Dev) | After (Stab) | Change |
|------|--------------|--------------|--------|
| Test Lead | 1.0 FTE | 0.5 FTE | -50% |
| Sr Engineers | 2.0 FTE | 1.5 FTE | -25% |
| Engineers | 2.0 FTE | 1.0 FTE | -50% |
| DevOps | 1.0 FTE | 1.0 FTE | - |
| Perf Engineer | 1.0 FTE | 1.0 FTE | - |
| QA | 1.0 FTE | 1.0 FTE | - |
| Tech Writer | 0.5 FTE | 0.5 FTE | - |
| **TOTAL** | **8.5 FTE** | **6.5 FTE** | **-23%** |

### Success Criteria
- ✅ All 7B models running on Linux Strix
- ✅ Memory constraints properly handled
- ✅ GPTQ quantization validated
- ✅ Test coverage: 105 test cases across 21 suites
- ✅ Weekly builds running smoothly

### Budget
**Weeks 1-16:** $655K cumulative

---

## M5: Full Integration Complete (Week 20)

### Objectives
- All 21 test suites integrated into CI/CD
- Automated weekly builds fully operational
- Performance monitoring dashboard live

### Deliverables

| # | Deliverable | Owner | Status |
|---|-------------|-------|--------|
| 1 | All 21 test suites in CI/CD | DevOps | □ |
| 2 | Automated test scheduling | DevOps | □ |
| 3 | Performance dashboard | Perf Engineer | □ |
| 4 | Regression detection system | Perf Engineer | □ |
| 5 | Automated alerting | DevOps | □ |
| 6 | Test execution runbooks | QA | □ |
| 7 | Troubleshooting guides | Tech Writer | □ |
| 8 | Strix-specific optimizations | Sr Engineer | □ |

### Integration Checklist

**CI/CD Pipeline:**
- [ ] All 21 test suites registered in `fetch_test_configurations.py`
- [ ] Test matrix generation working for gfx1151/gfx1150
- [ ] Platform-specific exclusions configured
- [ ] Artifact management functional
- [ ] Test result aggregation working
- [ ] Email notifications configured
- [ ] Slack integration (optional)

**Performance Monitoring:**
- [ ] Baseline metrics stored in database
- [ ] Trend analysis dashboard live
- [ ] Regression detection thresholds set
- [ ] Automated comparison (week-over-week)
- [ ] Performance reports auto-generated

**Documentation:**
- [ ] Architecture documentation complete
- [ ] API documentation for test framework
- [ ] Runbooks for all 21 test suites
- [ ] Troubleshooting guides
- [ ] Onboarding guide for new team members

### Success Criteria
- ✅ 4 consecutive weekly builds with 100% execution
- ✅ Performance dashboard operational
- ✅ Regression detection catching known issues
- ✅ Test execution time < 8 hours
- ✅ All documentation complete

### Budget
**Weeks 1-20:** $788K cumulative

### 🚦 GATE 3: Production Readiness Review

**Decision Point:** Ready for production release?

**Criteria:**
- [ ] All 21 test suites operational
- [ ] Weekly builds successful for 4 weeks
- [ ] No critical bugs outstanding
- [ ] Performance within targets
- [ ] Documentation complete
- [ ] Team comfortable with handoff to steady state

**Outcomes:**
- ✅ **GO:** Proceed to M6 (validation)
- ⚠️ **CONDITIONAL:** Address specific issues, then proceed
- ❌ **NO-GO:** Extended stabilization period required

---

## M6: Production Validation (Week 22)

### Objectives
- Validate all tests against acceptance criteria
- Complete end-to-end testing
- Sign-off from stakeholders

### Deliverables

| # | Deliverable | Owner | Status |
|---|-------------|-------|--------|
| 1 | Validation test report | QA | □ |
| 2 | Performance benchmark report | Perf Engineer | □ |
| 3 | Code coverage report (>90%) | Sr Engineer | □ |
| 4 | Security assessment | DevOps | □ |
| 5 | Acceptance test results | QA | □ |
| 6 | Stakeholder sign-off | Test Lead | □ |
| 7 | Production readiness checklist | Test Lead | □ |

### Validation Matrix

| Test Suite | Linux Pass | Windows Pass | Perf Target | Status |
|------------|------------|--------------|-------------|--------|
| qwen_vl | □ | □ | Latency <150ms | □ |
| vla_robotics | □ | N/A | FPS >30 | □ |
| multimodal_omni | □ | N/A | Latency <300ms | □ |
| sam2_segmentation | □ | □ | Latency <50ms | □ |
| llm_qwen_instruct | □ | □ | Latency <200ms | □ |
| asr_models | □ | □ | RTF <0.3 | □ |
| quantization_validation | □ | □ | Accuracy loss <2% | □ |
| strix_specific | □ | □ | All tests pass | □ |

### Acceptance Criteria

**Functional:**
- ✅ All tests pass on target platforms (100%)
- ✅ No critical/high severity bugs
- ✅ All error paths tested
- ✅ Edge cases handled gracefully

**Performance:**
- ✅ All performance targets met
- ✅ Test execution time within SLA (<8h)
- ✅ Memory usage within limits
- ✅ No performance regressions vs baseline

**Quality:**
- ✅ Code coverage ≥90%
- ✅ Documentation complete and reviewed
- ✅ Security scan passed
- ✅ No hardcoded credentials/secrets

### Success Criteria
- ✅ All validation tests pass
- ✅ Stakeholder sign-off received
- ✅ Production readiness checklist 100% complete
- ✅ Team trained and comfortable

### Budget
**Weeks 1-22:** $867K cumulative

---

## M7: Production Launch (Week 24)

### Objectives
- Transition to steady-state operations
- Hand off to maintenance team
- Establish operational excellence

### Deliverables

| # | Deliverable | Owner | Status |
|---|-------------|-------|--------|
| 1 | Weekly build schedule published | Test Lead | □ |
| 2 | On-call rotation established | DevOps | □ |
| 3 | Escalation procedures documented | Test Lead | □ |
| 4 | Knowledge transfer complete | All | □ |
| 5 | Post-launch review meeting | Test Lead | □ |
| 6 | Lessons learned document | Test Lead | □ |
| 7 | Continuous improvement plan | Test Lead | □ |

### Team Transition to Steady State

**Week 24:** Final transition

| Role | Before (Stab) | After (Steady) | Change |
|------|---------------|----------------|--------|
| Test Lead | 0.5 FTE | 0.25 FTE | -50% |
| Sr Engineers | 1.5 FTE | 1.0 FTE | -33% |
| Engineers | 1.0 FTE | 1.0 FTE | - |
| DevOps | 1.0 FTE | 0.5 FTE | -50% |
| Perf Engineer | 1.0 FTE | 0.5 FTE | -50% |
| QA | 1.0 FTE | 0.5 FTE | -50% |
| Tech Writer | 0.5 FTE | 0.0 FTE | -100% |
| **TOTAL** | **6.5 FTE** | **3.75 FTE** | **-42%** |

### Operational Processes

**Weekly Build Cycle:**
```
Monday 9am:    Build triggered
Monday 10am:   Tests start executing
Monday 6pm:    Results analyzed
Tuesday 9am:   Report published
Tuesday 10am:  Bug triage if needed
```

**Monthly Activities:**
- Performance trend review
- Infrastructure maintenance
- Dependency updates
- Team retrospective

**Quarterly Activities:**
- Add new model (1 per quarter)
- Architecture review
- Capacity planning
- Roadmap update

### Success Criteria
- ✅ First production weekly build successful
- ✅ Team operating independently
- ✅ All processes documented
- ✅ On-call coverage established
- ✅ Stakeholders satisfied

### Budget
**Weeks 1-24:** $946K cumulative

---

## Milestone Timeline (Gantt Chart)

```
Week:      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
──────────────────────────────────────────────────────────────────────────────────────────────────────────
M0: Kickoff        ◆
                   │
M1: Foundation     ├───────────────◆
                   │               │ GATE 1
M2: Phase 1        │               ├───────────────────────────◆
                   │               │                           │ GATE 2
M3: Phase 2        │               │                           ├───────────────────────────◆
                   │               │                           │                           │
M4: Phase 3        │               │                           │                           ├───────────────◆
                   │               │                           │                           │               │
M5: Integration    │               │                           │                           │               ├───────────────◆
                   │               │                           │                           │               │               │ GATE 3
M6: Validation     │               │                           │                           │               │               ├───────────◆
                   │               │                           │                           │               │               │           │
M7: Launch         │               │                           │                           │               │               │           ├───────◆
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Team Size:         8.5 FTE ────────────────────────────────────────────────────────►│  6.5 FTE ──────────►│  3.75 FTE
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Budget:            $42K/wk ────────────────────────────────────────────────────────►│  $32K/wk ──────────►│  $16K/wk
──────────────────────────────────────────────────────────────────────────────────────────────────────────
Cumulative:        $168K      $336K      $504K      $655K      $788K      $867K      $946K
──────────────────────────────────────────────────────────────────────────────────────────────────────────

Legend: ◆ = Milestone    │ = Work continues    ► = Transition
```

---

## Risk & Mitigation by Milestone

### M1 Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Hardware unavailable | Medium | High | Pre-order, have backup plan |
| Team not fully staffed | High | High | Start with contractors |
| ROCm issues | Medium | Medium | Have ROCm support engaged |

### M2 Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model licensing issues | Medium | Medium | Identify alternatives early |
| Framework scalability | Low | High | Load test early |
| Integration complexity | Medium | Medium | Allocate buffer time |

### M4 Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 7B model memory issues | High | Medium | Linux-only acceptable |
| Performance targets | Medium | Medium | Set realistic expectations |
| Team capacity | Medium | High | Prioritize ruthlessly |

### M5 Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CI/CD stability | Medium | High | Extensive testing before gate |
| Integration bugs | High | Medium | Allocate 4 weeks for fixes |
| Performance regressions | Medium | Medium | Have rollback plan |

---

## Budget by Milestone

| Milestone | Week | Duration | Weekly Cost | Milestone Cost | Cumulative |
|-----------|------|----------|-------------|----------------|------------|
| M0: Kickoff | 1 | 1 week | $42K | $42K | $42K |
| M1: Foundation | 1-4 | 3 weeks | $42K | $126K | $168K |
| M2: Phase 1 | 5-8 | 4 weeks | $42K | $168K | $336K |
| M3: Phase 2 | 9-12 | 4 weeks | $42K | $168K | $504K |
| M4: Phase 3 | 13-16 | 4 weeks | $38K | $151K | $655K |
| M5: Integration | 17-20 | 4 weeks | $32K | $133K | $788K |
| M6: Validation | 21-22 | 2 weeks | $32K | $79K | $867K |
| M7: Launch | 23-24 | 2 weeks | $32K | $79K | $946K |

**Total Program Cost:** $946K (6 months)

---

## Deliverables Summary by Milestone

| Milestone | Test Suites | Test Cases | Documentation | Major Features |
|-----------|-------------|------------|---------------|----------------|
| M0 | 0 | 0 | Charter, kickoff | Team formation |
| M1 | 1 | 1 | Architecture | Test framework, CI/CD |
| M2 | 4 | 28 | Test docs | Small models, weekly builds |
| M3 | 8 | 53 | Perf reports | Medium models, quantization |
| M4 | 14 | 105 | Full test docs | Large models, all categories |
| M5 | 21 | 105+ | Integration guides | Full automation, monitoring |
| M6 | 21 | 105+ | Validation reports | Acceptance testing |
| M7 | 21 | 105+ | Operations guides | Production launch |

---

## Key Performance Indicators (KPIs) by Milestone

### M1: Foundation
- [ ] CI/CD pipeline uptime: >95%
- [ ] Smoke test success rate: 100%
- [ ] Team velocity: On track

### M2: Phase 1
- [ ] Test pass rate: >90%
- [ ] Test execution time: <4 hours
- [ ] Weekly build success: 4/4

### M4: Phase 3
- [ ] All tests passing: 100%
- [ ] Code coverage: >85%
- [ ] Performance targets: 90% met

### M5: Integration
- [ ] Test automation: 85%
- [ ] Regression detection: Functional
- [ ] CI/CD stability: >98%

### M7: Launch
- [ ] Weekly build success: >95%
- [ ] Mean time to detect (MTTD): <24h
- [ ] Mean time to resolve (MTTR): <48h

---

## Decision Gates Summary

### 🚦 Gate 1: Foundation Complete (Week 4)
**Decision:** Continue to test development?  
**Criteria:** CI/CD working, smoke test passing, team on track  
**Risk if proceed without pass:** Cascading delays, tech debt

### 🚦 Gate 2: Phase 1 Complete (Week 8)
**Decision:** Proceed to medium/large models?  
**Criteria:** Small models working, weekly builds stable  
**Risk if proceed without pass:** Unstable foundation, quality issues

### 🚦 Gate 3: Production Readiness (Week 20)
**Decision:** Launch to production?  
**Criteria:** All tests working, performance validated, docs complete  
**Risk if proceed without pass:** Production incidents, rollback required

---

## Ongoing Steady State (Week 25+)

### Weekly Activities
- Monday: Build trigger, monitoring
- Tuesday: Result analysis, triage
- Wednesday: Bug fixes (if needed)
- Thursday: Performance review
- Friday: Team sync, planning

### Monthly Deliverables
- Performance trend report
- Test coverage report
- Incident summary
- Capacity analysis

### Quarterly Milestones
- Q1: Add 2-3 new models
- Q2: Optimization improvements
- Q3: Expand to new Strix variants
- Q4: Year-end review, planning

### Annual Budget (Steady State)
**Team:** 3.75 FTE + 20% contingency = 4.5 FTE  
**Cost:** $675K/year  
**Per Build:** $13K (52 builds/year)

---

## Success Metrics (Program-Level)

### On-Time Delivery
- **Target:** Deliver on Week 24
- **Buffer:** 2 weeks contingency built in
- **Track:** Weekly milestone progress

### Budget Adherence
- **Target:** ±10% of $946K
- **Track:** Weekly burn rate
- **Alert:** If >15% variance

### Quality
- **Target:** >90% test pass rate
- **Target:** >90% code coverage
- **Target:** Zero critical bugs at launch

### Team Health
- **Target:** <10% attrition
- **Track:** Weekly satisfaction surveys
- **Alert:** If velocity drops >20%

---

## Summary

**6-Month Program in 8 Milestones:**

| Phase | Weeks | Milestones | Team | Budget |
|-------|-------|------------|------|--------|
| **Development** | 1-16 | M0-M4 | 8.5 FTE → 6.5 FTE | $655K |
| **Stabilization** | 17-24 | M5-M7 | 6.5 FTE → 3.75 FTE | $291K |
| **TOTAL** | **24 weeks** | **8 milestones** | **8.5 → 3.75 FTE** | **$946K** |

**3 Decision Gates:** Weeks 4, 8, 20  
**Weekly Builds Start:** Week 8  
**Production Launch:** Week 24  
**Steady State:** 3.75-4.5 FTE, $675K/year

**Key Success Factors:**
- ✅ Clear milestone definitions
- ✅ Rigorous gate criteria
- ✅ Proactive risk management
- ✅ Regular stakeholder updates
- ✅ Team capacity planning

