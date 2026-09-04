<template>
  <div>
    <el-row :gutter="12" style="margin-bottom:16px;">
      <el-col :span="6" v-for="c in statusCards" :key="c.label">
        <el-card shadow="hover" :body-style="{padding:'12px 16px'}">
          <div style="display:flex;align-items:center;gap:10px;">
            <div :style="{width:'40px',height:'40px',borderRadius:'8px',background:c.bg,display:'flex',alignItems:'center',justifyContent:'center',fontSize:'18px'}">{{ c.icon }}</div>
            <div>
              <div style="font-size:22px;font-weight:700;color:#303133;">{{ c.value }}</div>
              <div style="font-size:12px;color:#909399;">{{ c.label }}</div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="block">
      <template #header>
        <div class="card-header">
          <span>巡检配置</span>
          <el-button size="small" type="primary" @click="saveConfig">保存</el-button>
        </div>
      </template>
      <el-form inline label-width="86px">
        <el-form-item label="提供商">
          <el-select v-model="form.provider" style="width:170px" @change="onProviderChange">
            <el-option v-for="p in providers" :key="p.name" :label="p.label" :value="p.name" />
          </el-select>
        </el-form-item>
        <el-form-item label="模型">
          <el-input v-model="form.model" style="width:200px" :placeholder="defaultModel" />
        </el-form-item>
        <el-form-item label="接口地址">
          <el-input v-model="form.base_url" style="width:270px" :placeholder="defaultBase" />
        </el-form-item>
        <el-form-item label="API Key">
          <el-input v-model="form.api_key" type="password" show-password style="width:210px" placeholder="留空使用环境变量" />
        </el-form-item>
        <el-form-item label="巡检间隔">
          <el-input-number v-model="form.interval_seconds" :min="10" :max="86400" />
        </el-form-item>
        <el-form-item label="温度">
          <el-input-number v-model="form.temperature" :min="0" :max="2" :step="0.1" />
        </el-form-item>
        <el-form-item label="最大输出">
          <el-input-number v-model="form.max_tokens" :min="64" :max="8192" :step="64" />
        </el-form-item>
        <el-form-item label="启用">
          <el-switch v-model="form.enabled" />
        </el-form-item>
      </el-form>
      <el-form label-width="86px">
        <el-form-item label="自定义提示词">
          <el-input v-model="form.prompt_override" type="textarea" :rows="2" placeholder="留空使用内置提示词（识别跌倒/姿态异常/地面杂物等）" />
        </el-form-item>
      </el-form>
    </el-card>

    <el-card shadow="never" class="block">
      <template #header><span>运行状态</span></template>
      <el-descriptions :column="4" size="small">
        <el-descriptions-item label="状态">{{ statusText }}</el-descriptions-item>
        <el-descriptions-item label="模型">{{ status.provider }} / {{ status.model || '-' }}</el-descriptions-item>
        <el-descriptions-item label="间隔">{{ status.interval_seconds }} 秒</el-descriptions-item>
        <el-descriptions-item label="最近执行">{{ fmtTime(status.last_run_at) }}</el-descriptions-item>
        <el-descriptions-item label="累计识别">{{ statsRuns }}</el-descriptions-item>
        <el-descriptions-item label="累计告警">{{ statsAlerts }}</el-descriptions-item>
        <el-descriptions-item label="累计错误">{{ statsErrors }}</el-descriptions-item>
        <el-descriptions-item label="最近错误">{{ status.last_error || '-' }}</el-descriptions-item>
      </el-descriptions>
    </el-card>

    <el-card shadow="never" class="block">
      <template #header>
        <div class="card-header">
          <span>立即检测</span>
          <div>
            <el-select v-model="targetId" style="width:220px" placeholder="选择设备">
              <el-option v-for="d in devices" :key="d.id" :label="d.device_name + '（' + d.scene + '）'" :value="d.id" />
            </el-select>
            <el-button type="primary" :loading="analyzing" @click="analyze">检测一次</el-button>
          </div>
        </div>
      </template>
      <div v-if="lastResult">
        <el-descriptions :column="3" size="small" border>
          <el-descriptions-item label="事件">{{ eventText(lastResult.event_type) }}</el-descriptions-item>
          <el-descriptions-item label="等级"><el-tag size="small" :type="levelType(lastResult.level)">{{ levelText(lastResult.level) }}</el-tag></el-descriptions-item>
          <el-descriptions-item label="置信度">{{ (lastResult.confidence * 100).toFixed(1) }}%</el-descriptions-item>
          <el-descriptions-item label="结论" :span="3">{{ lastResult.summary || '-' }}</el-descriptions-item>
          <el-descriptions-item label="细节" :span="3">{{ lastResult.details || '-' }}</el-descriptions-item>
          <el-descriptions-item label="建议" :span="3">{{ lastResult.suggestion || '-' }}</el-descriptions-item>
        </el-descriptions>
      </div>
      <el-empty v-else description="选择设备后点击“检测一次”" />
    </el-card>

    <el-card shadow="never" class="block">
      <template #header>
        <div class="card-header">
          <span>巡检历史</span>
          <div>
            <el-select v-model="filterEv" clearable placeholder="事件类型" style="width:150px" @change="loadResults">
              <el-option v-for="(t, k) in EVENT_TEXT" :key="k" :label="t" :value="k" />
            </el-select>
            <el-select v-model="filterSev" clearable placeholder="等级" style="width:120px" @change="loadResults">
              <el-option v-for="s in sevList" :key="s" :label="sevText(s)" :value="s" />
            </el-select>
            <el-button size="small" @click="loadResults">刷新</el-button>
          </div>
        </div>
      </template>
      <el-table :data="results" size="small" height="360">
        <el-table-column prop="id" label="ID" width="56" />
        <el-table-column label="设备" width="130">
          <template #default="s">{{ deviceName(s.row.device_id) }}</template>
        </el-table-column>
        <el-table-column label="事件" width="130">
          <template #default="s">{{ eventText(s.row.event_type) }}</template>
        </el-table-column>
        <el-table-column label="等级" width="90">
          <template #default="s"><el-tag size="small" :type="levelType(s.row.level)">{{ levelText(s.row.level) }}</el-tag></template>
        </el-table-column>
        <el-table-column label="置信度" width="90">
          <template #default="s">{{ (s.row.confidence * 100).toFixed(1) }}%</template>
        </el-table-column>
        <el-table-column prop="summary" label="结论" show-overflow-tooltip />
        <el-table-column label="时间" width="160">
          <template #default="s">{{ fmtTime(s.row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="耗时" width="80">
          <template #default="s">{{ s.row.latency_ms }}ms</template>
        </el-table-column>
        <el-table-column label="截图" width="70">
          <template #default="s">
            <el-button size="small" @click="preview(s.row)">查看</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog v-model="dialogVisible" title="巡检截图" width="720px">
      <img v-if="previewId" :src="imageUrl(previewId)" style="width:100%;" alt="巡检截图" />
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const EVENT_TEXT = { fall: '跌倒/倒地', posture_abnormal: '姿态/行为异常', floor_clutter: '地面杂物', other_risk: '其他风险', normal: '正常' }
const SEVERITY_TEXT = { low: '低', medium: '中', high: '高', critical: '危急' }
const sevList = ['low', 'medium', 'high', 'critical']

const providers = ref([])
const devices = ref([])
const results = ref([])
const status = ref({ enabled: false, running: false, stats: {} })
const form = reactive({ provider: 'qwen', model: '', base_url: '', api_key: '', interval_seconds: 60, temperature: 0.2, max_tokens: 800, enabled: false, prompt_override: '' })

const targetId = ref(null)
const analyzing = ref(false)
const lastResult = ref(null)
const filterEv = ref(null)
const filterSev = ref(null)
const dialogVisible = ref(false)
const previewId = ref(null)

const defaultProvider = computed(() => providers.value.find((p) => p.name === form.provider))
const defaultModel = computed(() => (defaultProvider.value && defaultProvider.value.default_model) || '')
const defaultBase = computed(() => (defaultProvider.value && defaultProvider.value.base_url) || '')
const statsRuns = computed(() => (status.value.stats && status.value.stats.runs) || 0)
const statsAlerts = computed(() => (status.value.stats && status.value.stats.alerts) || 0)
const statsErrors = computed(() => (status.value.stats && status.value.stats.errors) || 0)

const statusCards = computed(() => [
  { label: '巡检状态', value: statusText.value, icon: '\u25cf', bg: status.value.running ? '#e8f5e9' : '#f5f5f5' },
  { label: '累计识别', value: statsRuns.value, icon: '\u25a0', bg: '#e3f2fd' },
  { label: '累计告警', value: statsAlerts.value, icon: '\u25b2', bg: '#fff3e0' },
  { label: '累计错误', value: statsErrors.value, icon: '\u2716', bg: '#ffebee' },
])
const statusText = computed(() => {
  if (!status.value.enabled) return '未启用'
  return status.value.running ? '运行中' : '已停止'
})

function eventText(t) { return EVENT_TEXT[t] || t }
function sevText(s) { return SEVERITY_TEXT[s] || s }
function levelType(l) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[l] || 'info' }
function levelText(l) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[l] || l }
function deviceName(id) { const d = devices.value.find((x) => x.id === id); return d ? d.device_name : ('#' + id) }
function imageUrl(id) { return '/api/v1/multimodal/results/' + id + '/image' }
function fmtTime(v) {
  if (!v) return '-'
  const d = new Date(v)
  if (isNaN(d.getTime())) return String(v)
  const p = (n) => String(n).padStart(2, '0')
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' + p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds())
}

async function loadProviders() { providers.value = await api.multimodalProviders() }
async function loadConfig() {
  const c = await api.multimodalConfig()
  Object.assign(form, {
    provider: c.provider, model: c.model, base_url: c.base_url, api_key: '',
    interval_seconds: c.interval_seconds, temperature: c.temperature, max_tokens: c.max_tokens,
    enabled: c.enabled, prompt_override: c.prompt_override || ''
  })
}
async function loadStatus() { status.value = await api.multimodalStatus() }
async function loadDevices() { devices.value = await api.devices() }
async function loadResults() {
  const params = {}
  if (filterEv.value) params.event_type = filterEv.value
  if (filterSev.value) params.severity = filterSev.value
  results.value = await api.multimodalResults(params)
}
async function saveConfig() {
  await api.saveMultimodalConfig({ ...form })
  ElMessage.success('配置已保存')
  loadStatus()
}
async function analyze() {
  if (!targetId.value) { ElMessage.warning('请选择设备'); return }
  analyzing.value = true
  try {
    const r = await api.multimodalAnalyze(targetId.value)
    lastResult.value = r.result
    ElMessage.success('检测完成')
    loadResults()
  } catch (e) {
    ElMessage.error((e.response && e.response.data && e.response.data.detail) || '检测失败')
    lastResult.value = null
  } finally {
    analyzing.value = false
  }
}
function preview(row) { previewId.value = row.id; dialogVisible.value = true }
function onProviderChange() { form.model = ''; form.base_url = '' }

onMounted(() => { loadProviders(); loadConfig(); loadStatus(); loadDevices(); loadResults() })
</script>

<style scoped>
.block { margin-bottom: 16px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>
