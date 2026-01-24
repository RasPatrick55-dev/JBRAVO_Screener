---
name: Dashboard Task
about: UI, data, or bug work for the JBRAVO dashboard
title: ''
labels: dashboard-bug, dashboard-data, dashboard-enhancement, dashboard-tech-debt,
  dashboard-ui
assignees: RasPatrick55-dev

---

name: Dashboard Task
description: UI, data, or bug work for the JBRAVO dashboard
title: "[Dashboard] <short description>"
labels: ["dashboard-enhancement"]
body:
  - type: dropdown
    id: task_type
    attributes:
      label: Task Type
      options:
        - UI / UX
        - Data / SQL
        - Bug
        - Performance
        - Tech Debt
    validations:
      required: true

  - type: input
    id: dashboard_tab
    attributes:
      label: Dashboard Tab(s)
      placeholder: "Screener Health, Execution, Trades, Overview"
    validations:
      required: true

  - type: textarea
    id: problem
    attributes:
      label: Problem / Goal
    validations:
      required: true

  - type: textarea
    id: user_value
    attributes:
      label: User Value
    validations:
      required: true

  - type: textarea
    id: data_source
    attributes:
      label: Data Source
      placeholder: "PostgreSQL table/view or CSV fallback"
    validations:
      required: true

  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance Criteria
      placeholder: |
        - Loads with no data
        - Loads with data
        - Correct KPIs
        - No console errors
    validations:
      required: true

  - type: textarea
    id: figma
    attributes:
      label: Figma Link (added later)
