-- Orchestration multi-agents (mémoire partagée).
-- N'altère PAS les tables scientifiques (features_*, cours, fondamentaux).
-- Exécuter dans Supabase SQL Editor (rôle postgres). Résultats = append-only via run_id.

CREATE TABLE IF NOT EXISTS public.agent_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id uuid,
    agent_name text NOT NULL,
    task_type text NOT NULL,
    period_start text,
    period_end text,
    configuration text,
    status text NOT NULL DEFAULT 'PENDING',
    started_at timestamptz,
    completed_at timestamptz,
    error_message text,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_jobs_workflow ON public.agent_jobs (workflow_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_jobs_status ON public.agent_jobs (agent_name, status);

CREATE TABLE IF NOT EXISTS public.agent_logs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id uuid REFERENCES public.agent_jobs (id) ON DELETE CASCADE,
    workflow_id uuid,
    agent_name text NOT NULL,
    level text NOT NULL DEFAULT 'INFO',
    message text NOT NULL,
    payload jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_logs_job ON public.agent_logs (job_id, created_at);

CREATE TABLE IF NOT EXISTS public.agent_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id uuid,
    job_id uuid,
    event_type text NOT NULL,
    producer_agent text NOT NULL,
    payload jsonb DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_events_wf ON public.agent_events (workflow_id, created_at);

CREATE TABLE IF NOT EXISTS public.agent_result_registry (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL,
    agent_name text NOT NULL,
    configuration text,
    period_start text,
    period_end text,
    model_version text,
    parameters jsonb DEFAULT '{}'::jsonb,
    artifact_path text,
    checksum_sha256 text,
    status text NOT NULL,
    produced_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_recommendations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL,
    agent_name text NOT NULL DEFAULT 'selection',
    cell int NOT NULL,
    configuration text,
    month text NOT NULL,
    date_decision date,
    ticker text NOT NULL,
    recommendation text NOT NULL,
    score double precision,
    conviction_score double precision,
    tau double precision,
    model_version text,
    parameters jsonb DEFAULT '{}'::jsonb,
    produced_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'SUCCESS'
);

CREATE INDEX IF NOT EXISTS idx_agent_rec_run ON public.agent_recommendations (run_id, cell, month);

CREATE TABLE IF NOT EXISTS public.agent_portfolio_holdings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL,
    agent_name text NOT NULL DEFAULT 'allocation',
    cell int NOT NULL,
    configuration text,
    month text NOT NULL,
    date_rebalance date,
    ticker text NOT NULL,
    weight double precision NOT NULL,
    recommendation text,
    predicted_return double precision,
    model_version text,
    parameters jsonb DEFAULT '{}'::jsonb,
    produced_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'SUCCESS'
);

CREATE INDEX IF NOT EXISTS idx_agent_hold_run ON public.agent_portfolio_holdings (run_id, cell, month);

CREATE TABLE IF NOT EXISTS public.agent_backtest_monthly (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL,
    agent_name text NOT NULL DEFAULT 'backtest',
    cell int NOT NULL,
    configuration text,
    month text NOT NULL,
    net_return double precision,
    gross_return double precision,
    turnover double precision,
    transaction_cost double precision,
    n_positions int,
    wealth100 double precision,
    model_version text,
    parameters jsonb DEFAULT '{}'::jsonb,
    produced_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'SUCCESS'
);

CREATE TABLE IF NOT EXISTS public.agent_evaluation_kpis (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL,
    agent_name text NOT NULL DEFAULT 'evaluation',
    configuration text,
    strategie text NOT NULL,
    n_months int,
    rendement_ann double precision,
    volatilite double precision,
    sharpe double precision,
    sortino double precision,
    cvar double precision,
    max_dd double precision,
    turnover double precision,
    liquidite double precision,
    wealth_final_100 double precision,
    model_version text,
    parameters jsonb DEFAULT '{}'::jsonb,
    produced_at timestamptz NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'SUCCESS'
);

ALTER TABLE public.agent_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_result_registry ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_portfolio_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_backtest_monthly ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_evaluation_kpis ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'agent_jobs','agent_logs','agent_events','agent_result_registry',
    'agent_recommendations','agent_portfolio_holdings',
    'agent_backtest_monthly','agent_evaluation_kpis'
  ]
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS agent_select_anon ON public.%I', t);
    EXECUTE format(
      'CREATE POLICY agent_select_anon ON public.%I FOR SELECT TO anon, authenticated USING (true)',
      t
    );
    EXECUTE format('DROP POLICY IF EXISTS agent_write_service ON public.%I', t);
    EXECUTE format(
      'CREATE POLICY agent_write_service ON public.%I FOR ALL TO authenticated USING (true) WITH CHECK (true)',
      t
    );
  END LOOP;
END $$;

GRANT SELECT, INSERT, UPDATE ON
  public.agent_jobs, public.agent_logs, public.agent_events,
  public.agent_result_registry, public.agent_recommendations,
  public.agent_portfolio_holdings, public.agent_backtest_monthly,
  public.agent_evaluation_kpis
TO anon, authenticated;
