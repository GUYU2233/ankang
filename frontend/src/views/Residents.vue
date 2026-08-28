<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>老人档案</span>
        <div>
          <el-button size="small" type="primary" @click="load">刷新</el-button>
          <el-button size="small" type="success" @click="openAdd">新增档案</el-button>
        </div>
      </div>
    </template>
    <el-table :data="residents" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" label="姓名" width="110" />
      <el-table-column prop="gender" label="性别" width="65" />
      <el-table-column prop="age" label="年龄" width="65" />
      <el-table-column prop="guardian_name" label="家属/护工" width="110" />
      <el-table-column prop="guardian_phone" label="联系电话" width="135" />
      <el-table-column label="名下设备" width="170">
        <template #default="s">
          <template v-if="(devicesByResident[s.row.id] || []).length">
            <el-tag v-for="d in devicesByResident[s.row.id]" :key="d.id" size="small" style="margin: 2px 4px 2px 0;">{{ d.device_name + '·' + d.scene }}</el-tag>
          </template>
          <span v-else>-</span>
        </template>
      </el-table-column>
      <el-table-column prop="address" label="住址" />
      <el-table-column prop="medical_history" label="病史" show-overflow-tooltip />
      <el-table-column label="操作" width="110">
        <template #default="s">
          <el-button size="small" type="primary" @click="openRiskProfile(s.row)">风险档案</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="dialogVisible" title="老人档案" width="520px">
      <el-form :model="form" label-width="100px">
        <el-form-item label="姓名"><el-input v-model="form.name" /></el-form-item>
        <el-form-item label="性别"><el-input v-model="form.gender" /></el-form-item>
        <el-form-item label="年龄"><el-input v-model="form.age" type="number" /></el-form-item>
        <el-form-item label="家属姓名"><el-input v-model="form.guardian_name" /></el-form-item>
        <el-form-item label="家属电话"><el-input v-model="form.guardian_phone" /></el-form-item>
        <el-form-item label="住址"><el-input v-model="form.address" /></el-form-item>
        <el-form-item label="病史"><el-input v-model="form.medical_history" type="textarea" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="onAdd">确定</el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="drawerVisible" size="640px" :title="(currentResident?.name || '') + ' · 风险档案'" @opened="initCharts">
      <div v-if="profile">
        <el-descriptions :column="1" border size="small">
          <el-descriptions-item label="最新风险分">{{ profile.latest_score != null ? Number(profile.latest_score).toFixed(2) : '-' }}</el-descriptions-item>
          <el-descriptions-item label="风险等级">
            <el-tag v-if="profile.latest_level" :type="levelType(profile.latest_level)" size="small">{{ levelText(profile.latest_level) }}</el-tag>
            <span v-else>-</span>
          </el-descriptions-item>
          <el-descriptions-item label="评估时间">{{ profile.updated_at || '-' }}</el-descriptions-item>
        </el-descriptions>
        <div class="chart-title">近 7 天风险趋势</div>
        <div v-if="trend.length" ref="lineEl" class="chart-box"></div>
        <el-empty v-else description="暂无趋势数据" :image-size="60" />
        <div class="chart-title">风险因子雷达</div>
        <div v-if="factors.length" ref="radarEl" class="chart-box"></div>
        <el-empty v-else description="暂无因子数据" :image-size="60" />
      </div>
      <el-empty v-else description="暂无数据" :image-size="80" />
    </el-drawer>
  </el-card>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import api from '../api'

const residents = ref([])
const devices = ref([])
const dialogVisible = ref(false)
const drawerVisible = ref(false)
const currentResident = ref(null)
const profile = ref(null)
const lineEl = ref(null)
const radarEl = ref(null)
const form = reactive({ name: '', gender: '男', age: null, guardian_name: '', guardian_phone: '', address: '', medical_history: '' })

let lineChart = null
let radarChart = null

const devicesByResident = computed(() => {
  const map = {}
  for (const d of devices.value) {
    if (!d.resident_id) continue
    if (!map[d.resident_id]) map[d.resident_id] = []
    map[d.resident_id].push(d)
  }
  return map
})
const trend = computed(() => {
  const t = profile.value?.trend
  return Array.isArray(t) ? t : []
})
const factors = computed(() => {
  const f = profile.value?.factors
  return Array.isArray(f) ? f : []
})

async function load() {
  residents.value = await api.residents()
}
function openAdd() {
  Object.assign(form, { name: '', gender: '男', age: null, guardian_name: '', guardian_phone: '', address: '', medical_history: '' })
  dialogVisible.value = true
}
async function onAdd() {
  await api.addResident(form)
  ElMessage.success('档案已保存')
  dialogVisible.value = false
  load()
}

function levelType(level) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[level] || 'info' }
function levelText(level) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[level] || level }

function disposeCharts() {
  if (lineChart) { lineChart.dispose(); lineChart = null }
  if (radarChart) { radarChart.dispose(); radarChart = null }
}

function initCharts() {
  disposeCharts()
  const lineElv = lineEl.value
  const radarElv = radarEl.value
  if (lineElv && trend.value.length) {
    lineChart = echarts.init(lineElv)
    lineChart.setOption({
      tooltip: { trigger: 'axis' },
      grid: { left: 40, right: 20, top: 30, bottom: 30 },
      xAxis: { type: 'category', data: trend.value.map(p => p.date) },
      yAxis: { type: 'value', min: 0, max: 1 },
      series: [{ name: '平均风险分', type: 'line', smooth: true, areaStyle: {}, data: trend.value.map(p => p.avg_score) }]
    })
  }
  if (radarElv && factors.value.length) {
    radarChart = echarts.init(radarElv)
    const values = factors.value.map(f => Number(f.value) || 0)
    const max = Math.max(1, ...values)
    radarChart.setOption({
      tooltip: {},
      radar: {
        indicator: factors.value.map(f => ({ name: f.label, max })),
        radius: '65%'
      },
      series: [{
        type: 'radar',
        data: [{ value: values, name: '风险因子' }]
      }]
    })
  }
}

async function openRiskProfile(row) {
  currentResident.value = row
  profile.value = null
  drawerVisible.value = true
  try {
    const data = await api.residentRiskProfile(row.id, 7)
    profile.value = data || {}
  } catch (e) {
    profile.value = {}
  }
  await nextTick()
  initCharts()
}

function onResize() {
  if (lineChart) lineChart.resize()
  if (radarChart) radarChart.resize()
}

watch(drawerVisible, (v) => { if (!v) disposeCharts() })

onMounted(() => {
  load()
  loadDevices()
  window.addEventListener('resize', onResize)
})
onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  disposeCharts()
})

async function loadDevices() {
  try { devices.value = await api.devices() } catch (e) { /* ignore */ }
}
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
.chart-title { margin: 16px 0 8px; font-weight: 600; color: #303133; }
.chart-box { height: 280px; width: 100%; }
</style>
