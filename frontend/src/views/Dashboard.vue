<template>
  <div>
    <!-- 告警提示音（可自定义音源） -->
    <audio ref="alertAudio" :src="alertSoundUrl" preload="auto" />

    <el-row :gutter="16">
      <el-col :span="6" v-for="c in cards" :key="c.label">
        <el-card shadow="never">
          <div class="stat-value">{{ c.value }}</div>
          <div class="stat-label">{{ c.label }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>最新告警</span>
              <el-button size="small" @click="loadStats">刷新</el-button>
            </div>
          </template>
          <el-table :data="alerts" size="small" height="340">
            <el-table-column prop="level" label="级别" width="90">
              <template #default="s">
                <el-tag :type="levelType(s.row.level)" size="small">{{ levelText(s.row.level) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="title" label="内容" />
            <el-table-column prop="created_at" label="时间" width="180" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="10">
        <el-card shadow="never">
          <template #header><span>系统状态</span></template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="后端服务"><el-tag :type="online ? 'success' : 'danger'" size="small">{{ online ? '在线' : '离线' }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="WebSocket" ><el-tag :type="wsOpen ? 'success' : 'danger'" size="small">{{ wsOpen ? '已连接' : '未连接' }}</el-tag></el-descriptions-item>
            <el-descriptions-item label="设备总数">{{ stats.total_devices }}</el-descriptions-item>
            <el-descriptions-item label="在线设备">{{ stats.online_devices }}</el-descriptions-item>
            <el-descriptions-item label="老人档案">{{ stats.total_residents }}</el-descriptions-item>
            <el-descriptions-item label="今日跌倒告警">{{ stats.today_falls }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="24">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>系统设置：告警通知 &amp; 提示音</span>
              <el-button size="small" @click="showSettings = !showSettings">{{ showSettings ? '收起' : '展开' }}</el-button>
            </div>
          </template>
          <div v-if="showSettings" style="display:flex;gap:24px;flex-wrap:wrap;">
            <!-- 提示音设置 -->
            <div style="flex:1;min-width:280px;">
              <h4>跌倒提示音</h4>
              <el-input v-model="alertSoundUrl" placeholder="提示音 URL（mp3/wav）" size="small">
                <template #append><el-button size="small" @click="saveSoundUrl">保存</el-button></template>
              </el-input>
              <el-button size="small" style="margin-top:6px;" @click="alertAudio && alertAudio.play().catch(()=>{})">试听</el-button>
              <el-button size="small" style="margin-top:6px;" @click="alertSoundUrl='https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3';saveSoundUrl()">恢复默认</el-button>
            </div>
            <!-- Webhook 渠道 -->
            <div style="flex:2;min-width:400px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <h4 style="margin:0;">消息推送渠道</h4>
                <el-button size="small" type="primary" @click="openWebhookForm()">添加</el-button>
              </div>
              <el-table :data="webhooks" size="small" max-height="200">
                <el-table-column prop="name" label="名称" width="100" />
                <el-table-column label="平台" width="80">
                  <template #default="s">{{ {wechat:'微信',dingtalk:'钉钉',feishu:'飞书',custom:'自定义'}[s.row.platform] || s.row.platform }}</template>
                </el-table-column>
                <el-table-column prop="webhook_url_masked" label="地址" show-overflow-tooltip />
                <el-table-column prop="trigger_levels" label="触发等级" width="90" />
                <el-table-column label="启用" width="60">
                  <template #default="s"><el-tag :type="s.row.enabled?'success':'info'" size="small">{{ s.row.enabled?'是':'否' }}</el-tag></template>
                </el-table-column>
                <el-table-column label="操作" width="140">
                  <template #default="s">
                    <el-button size="small" @click="editWebhook(s.row)">编辑</el-button>
                    <el-button size="small" @click="testWebhook(s.row.id)">测试</el-button>
                    <el-button size="small" type="danger" @click="deleteWebhook(s.row.id)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
              <!-- Webhook 编辑表单 -->
              <el-form v-if="showWebhookForm" inline size="small" style="margin-top:8px;background:#fafafa;padding:8px;border-radius:4px;">
                <el-form-item label="名称"><el-input v-model="webhookForm.name" size="small" style="width:120px" placeholder="如：家属钉钉群" /></el-form-item>
                <el-form-item label="平台">
                  <el-select v-model="webhookForm.platform" size="small" style="width:100px">
                    <el-option label="钉钉" value="dingtalk" /><el-option label="飞书" value="feishu" /><el-option label="企业微信" value="wechat" /><el-option label="自定义" value="custom" />
                  </el-select>
                </el-form-item>
                <el-form-item label="Webhook URL"><el-input v-model="webhookForm.webhook_url" size="small" style="width:240px" placeholder="https://..." /></el-form-item>
                <el-form-item label="密钥"><el-input v-model="webhookForm.secret" size="small" style="width:130px" placeholder="签名密钥（可选）" /></el-form-item>
                <el-form-item label="触发等级"><el-input v-model="webhookForm.trigger_levels" size="small" style="width:100px" placeholder="red,orange" /></el-form-item>
                <el-form-item label="启用"><el-switch v-model="webhookForm.enabled" size="small" /></el-form-item>
                <el-form-item><el-button size="small" type="primary" @click="saveWebhook">保存</el-button></el-form-item>
              </el-form>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="16" style="margin-top: 16px;">
      <el-col :span="24">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>近7天风险趋势</span>
              <el-button size="small" @click="loadTrend">刷新</el-button>
            </div>
          </template>
          <div ref="trendEl" class="trend-box"></div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { computed, onMounted, onBeforeUnmount, reactive, ref } from 'vue'
import * as echarts from 'echarts'
import { ElMessage, ElNotification } from 'element-plus'
import api from '../api'

const stats = ref({ total_devices: 0, online_devices: 0, total_residents: 0, today_alerts: 0, today_falls: 0, avg_risk_score: 0, latest_alerts: [] })
const online = ref(false)
const wsOpen = ref(false)
const trendEl = ref(null)
const alertAudio = ref(null)
let ws = null
let timer = null
let trendChart = null

// 告警提示音设置（localStorage 持久化）
const alertSoundUrl = ref(localStorage.getItem('ankang_alert_sound') || 'https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3')
const showSettings = ref(false)
const showWebhookForm = ref(false)
const webhooks = ref([])
const webhookForm = reactive({ name: '', platform: 'custom', webhook_url: '', secret: '', enabled: true, trigger_levels: 'red,orange' })
const webhookEditing = ref(null)

function saveSoundUrl() {
  localStorage.setItem('ankang_alert_sound', alertSoundUrl.value)
  ElMessage.success('提示音已保存')
}
async function loadWebhooks() {
  try { webhooks.value = await api.webhooks() } catch (_) {}
}
async function saveWebhook() {
  try {
    if (webhookEditing.value) {
      await api.updateWebhook(webhookEditing.value, { ...webhookForm })
    } else {
      await api.createWebhook({ ...webhookForm })
    }
    ElMessage.success('已保存')
    webhookForm.name = ''; webhookForm.platform = 'custom'; webhookForm.webhook_url = ''; webhookForm.secret = ''; webhookForm.enabled = true; webhookForm.trigger_levels = 'red,orange'
    webhookEditing.value = null
    showWebhookForm.value = false
    loadWebhooks()
  } catch (e) { ElMessage.error(e.response?.data?.detail || '保存失败') }
}
function openWebhookForm() {
  webhookEditing.value = null
  showWebhookForm.value = true
  Object.assign(webhookForm, { name: '', platform: 'custom', webhook_url: '', secret: '', enabled: true, trigger_levels: 'red,orange' })
}
function editWebhook(w) {
  webhookEditing.value = w.id
  showWebhookForm.value = true
  Object.assign(webhookForm, { name: w.name, platform: w.platform, webhook_url: '', secret: '', enabled: w.enabled, trigger_levels: w.trigger_levels })
}
async function deleteWebhook(id) {
  try { await api.deleteWebhook(id); ElMessage.success('已删除'); loadWebhooks() } catch (_) {}
}
async function testWebhook(id) {
  try { const r = await api.testWebhook(id); ElMessage[r.ok ? 'success' : 'error'](r.detail) } catch (_) {}
}

const cards = computed(() => [
  { label: '设备总数', value: stats.value.total_devices },
  { label: '在线设备', value: stats.value.online_devices },
  { label: '今日告警', value: stats.value.today_alerts },
  { label: '平均风险评分', value: stats.value.avg_risk_score }
])
const alerts = computed(() => stats.value.latest_alerts || [])

function levelType(level) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[level] || 'info' }
function levelText(level) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[level] || level }

async function loadStats() {
  try {
    const data = await api.dashboard()
    stats.value = data
    online.value = true
  } catch (e) {
    online.value = false
  }
}

async function loadTrend() {
  try {
    const data = await api.riskTrend(7)
    const trend = Array.isArray(data) ? data : (data.trend || [])
    if (!trendEl.value) return
    if (!trendChart) trendChart = echarts.init(trendEl.value)
    trendChart.setOption({
      tooltip: { trigger: 'axis' },
      legend: { data: ['平均风险分', '峰值'] },
      grid: { left: 40, right: 20, top: 40, bottom: 30 },
      xAxis: { type: 'category', data: trend.map(p => p.date) },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: [
        { name: '平均风险分', type: 'line', smooth: true, areaStyle: {}, data: trend.map(p => p.avg_score) },
        { name: '峰值', type: 'line', smooth: true, lineStyle: { type: 'dashed' }, data: trend.map(p => p.max_score) }
      ]
    })
  } catch (e) { /* 趋势加载失败不打断看板 */ }
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  ws = new WebSocket(proto + '://' + location.host + '/ws/alerts')
  ws.onopen = () => { wsOpen.value = true }
  ws.onclose = () => { wsOpen.value = false }
  ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data)
      if (msg.type === 'alert') {
        const d = msg.data || {}
        const name = d.resident_name ? d.resident_name + ' · ' : ''
        const msgText = name + (d.title || '') + ' [' + (d.scene || '') + ']' + (d.guardian_phone ? '（家属: ' + d.guardian_phone + '）' : '')
        ElNotification({ title: '收到预警', message: msgText, type: d.level === 'red' ? 'error' : 'warning', duration: 6000 })
        if (d.level === 'red' || d.level === 'orange') {
          try { alertAudio.value && alertAudio.value.play().catch(() => {}) } catch (_) {}
        }
        loadStats()
        loadTrend()
      }
    } catch (e) { /* ignore */ }
  }
}

function onResize() {
  if (trendChart) trendChart.resize()
}

onMounted(() => {
  loadStats()
  loadTrend()
  connectWs()
  loadWebhooks()
  timer = setInterval(loadStats, 5000)
  window.addEventListener('resize', onResize)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
  if (ws) ws.close()
  window.removeEventListener('resize', onResize)
  if (trendChart) { trendChart.dispose(); trendChart = null }
})
</script>

<style scoped>
.stat-value { font-size: 30px; font-weight: 700; color: #303133; }
.stat-label { color: #909399; margin-top: 6px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.trend-box { height: 300px; width: 100%; }
</style>