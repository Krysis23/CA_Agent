import { motion } from 'framer-motion';
import { Building2, Calendar, TrendingUp, TrendingDown, Wallet, ArrowUpDown, Receipt, IndianRupee } from 'lucide-react';

export interface BankSummaryData {
  personEntity: string;
  period: string;
  openingBalance: string;
  totalCredits: string;
  totalDebits: string;
  closingBalance: string;
  estimatedAnnualIncome: string;
  estimatedTax: string;
}

interface BankStatementCardProps {
  data: BankSummaryData;
  filename?: string;
}

const StatRow = ({
  icon: Icon,
  label,
  value,
  colorClass,
  bgClass,
  delay,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  colorClass: string;
  bgClass: string;
  delay: number;
}) => (
  <motion.div
    initial={{ opacity: 0, x: -12 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.35, delay }}
    className={`flex items-center justify-between rounded-xl px-4 py-3 ${bgClass} border border-white/5`}
  >
    <div className="flex items-center gap-3">
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colorClass} bg-current/10`}>
        <Icon size={15} className="text-current opacity-80" style={{ color: 'inherit' }} />
      </div>
      <span className="text-sm text-muted-foreground font-medium">{label}</span>
    </div>
    <span className={`text-sm font-bold tabular-nums ${colorClass}`}>{value}</span>
  </motion.div>
);

const BankStatementCard = ({ data, filename }: BankStatementCardProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="mt-3 rounded-2xl border border-border bg-card overflow-hidden shadow-lg"
    >
      {/* Header */}
      <div className="px-5 py-4 bg-gradient-to-r from-primary/20 via-primary/10 to-transparent border-b border-border flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl gradient-primary flex items-center justify-center shrink-0 shadow-md">
            <Building2 size={18} className="text-primary-foreground" />
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-primary mb-0.5">
              Bank Statement Summary
            </p>
            <p className="text-sm font-bold text-foreground leading-tight">
              {data.personEntity !== 'N/A' ? data.personEntity : filename ?? 'Statement'}
            </p>
          </div>
        </div>

        {data.period !== 'N/A' && (
          <div className="flex items-center gap-1.5 bg-muted/60 rounded-lg px-2.5 py-1.5 shrink-0">
            <Calendar size={12} className="text-muted-foreground" />
            <span className="text-[11px] text-muted-foreground font-medium">{data.period}</span>
          </div>
        )}
      </div>

      {/* Stats grid */}
      <div className="p-4 space-y-2.5">
        <StatRow
          icon={Wallet}
          label="Opening Balance"
          value={data.openingBalance}
          colorClass="text-foreground"
          bgClass="bg-muted/30"
          delay={0.05}
        />
        <StatRow
          icon={TrendingUp}
          label="Total Credits"
          value={data.totalCredits}
          colorClass="text-emerald-400"
          bgClass="bg-emerald-500/8"
          delay={0.1}
        />
        <StatRow
          icon={TrendingDown}
          label="Total Debits"
          value={data.totalDebits}
          colorClass="text-rose-400"
          bgClass="bg-rose-500/8"
          delay={0.15}
        />

        {/* Divider */}
        <div className="border-t border-border/60 my-1" />

        <StatRow
          icon={ArrowUpDown}
          label="Closing Balance"
          value={data.closingBalance}
          colorClass="text-primary"
          bgClass="bg-primary/8"
          delay={0.2}
        />
      </div>

      {/* Tax Estimate Section */}
      {data.estimatedTax && data.estimatedTax !== 'N/A' && data.estimatedTax !== '₹0' && (
        <div className="border-t border-border">
          <div className="px-4 py-2.5 bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-amber-400/80">
              Tax Estimate · New Regime FY 2025-26
            </p>
          </div>
          <div className="px-4 pb-4 pt-1 space-y-2.5">
            <StatRow
              icon={IndianRupee}
              label="Estimated Annual Income"
              value={data.estimatedAnnualIncome}
              colorClass="text-amber-400"
              bgClass="bg-amber-500/8"
              delay={0.25}
            />
            <StatRow
              icon={Receipt}
              label="Estimated Tax (incl. Cess)"
              value={data.estimatedTax}
              colorClass="text-amber-300"
              bgClass="bg-amber-500/8"
              delay={0.3}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default BankStatementCard;
