<template>
  <el-card shadow="never">
    <template #header>
      <div class="card-header">
        <span>告警中心</span>
        <div>
          <el-radio-group v-model="filterLevel" size="small" @change="load">
            <el-radio-button label="">全部</el-radio-button>
            <el-radio-button label="red">红色</el-radio-button>
            <el-radio-button label="orange">橙色</el-radio-button>
            <el-radio-button label="yellow">黄色</el-radio-button>
          </el-radio-group>
          <el-radio-group v-model="filterStatus" size="small" style="margin-left:8px;" @change="load">
            <el-radio-button label="">全部状态</el-radio-button>
            <el-radio-button label="pending">待确认</el-radio-button>
            <el-radio-button label="confirmed">已确认</el-radio-button>
            <el-radio-button label="handled">已处置</el-radio-button>
            <el-radio-button label="closed">已归档</el-radio-button>
          </el-radio-group>
          <el-button size="small" type="primary" @click="load" style="margin-left:8px;">刷新</el-button>
        </div>
      </div>
    </template>
    <el-table :data="alerts" size="small">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="level" label="级别" width="100">
        <template #default="s">
          <el-tag :type="levelType(s.row.level)" size="small">{{ levelText(s.row.level) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="title" label="内容" />
      <el-table-column prop="event_type" label="类型" width="120" />
      <el-table-column label="状态" width="90">
        <template #default="s">
          <el-tag :type="statusType(statusOf(s.row))" size="small">{{ statusText(statusOf(s.row)) }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="归属老人" width="110">
        <template #default="s">{{ s.row.resident_name || '-' }}</template>
      </el-table-column>
      <el-table-column label="联系电话" width="130">
        <template #default="s">{{ s.row.guardian_phone || '-' }}</template>
      </el-table-column>
      <el-table-column label="处理人" width="100">
        <template #default="s">{{ s.row.handled_by || s.row.confirmed_by || '-' }}</template>
      </el-table-column>
      <el-table-column prop="created_at" label="时间" width="180" />
      <el-table-column label="操作" width="100">
        <template #default="s">
          <el-button v-if="statusOf(s.row) === 'pending'" size="small" type="primary" @click="openAction(s.row, 'confirm')">确认</el-button>
          <el-button v-else-if="statusOf(s.row) === 'confirmed'" size="small" type="warning" @click="openAction(s.row, 'handle')">处置</el-button>
          <el-button v-else-if="statusOf(s.row) === 'handled'" size="small" type="info" @click="openAction(s.row, 'close')">归档</el-button>
          <el-tag v-else size="small" type="info">已归档</el-tag>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="actionVisible" :title="actionTitle" width="480px">
      <el-form :model="actionForm" label-width="80px">
        <el-form-item label="操作人">
          <el-input v-model="actionForm.operator" placeholder="操作人姓名" />
        </el-form-item>
        <el-form-item label="备注">
          <el-input v-model="actionForm.note" type="textarea" :rows="3" placeholder="备注信息（可选）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="actionVisible = false">取消</el-button>
        <el-button type="primary" @click="submitAction">提交</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const alerts = ref([])
const filterLevel = ref('')
const filterStatus = ref('')

const actionVisible = ref(false)
const action = ref('confirm')
const actionRow = ref(null)
const actionForm = reactive({ operator: '值班员', note: '' })
const actionTitle = computed(() => ({ confirm: '确认告警', handle: '处置告警', close: '归档告警' }[action.value] || '处置告警'))

function levelType(level) { return { red: 'danger', orange: 'warning', yellow: 'warning', green: 'success' }[level] || 'info' }
function levelText(level) { return { red: '红色紧急', orange: '橙色高风险', yellow: '黄色关注', green: '绿色正常' }[level] || level }
function statusOf(row) {
  if (row.status) return row.status
  if (row.confirmed) return row.handled ? 'handled' : 'confirmed'
  return 'pending'
}
function statusType(st) { return { pending: 'warning', confirmed: 'primary', handled: 'success', closed: 'info' }[st] || 'info' }
function statusText(st) { return { pending: '待确认', confirmed: '已确认', handled: '已处置', closed: '已归档' }[st] || st }

async function load() {
  alerts.value = await api.alerts(filterLevel.value || null, filterStatus.value || null)
}

function openAction(row, act) {
  action.value = act
  actionRow.value = row
  actionForm.operator = '值班员'
  actionForm.note = ''
  actionVisible.value = true
}

async function submitAction() {
  if (!actionRow.value) return
  const id = actionRow.value.id
  const payload = { operator: actionForm.operator || '值班员', note: actionForm.note }
  try {
    if (action.value === 'confirm') await api.confirmAlert(id, payload)
    else if (action.value === 'handle') await api.handleAlert(id, payload)
    else await api.closeAlert(id, payload)
    ElMessage.success(action.value === 'confirm' ? '已确认' : action.value === 'handle' ? '已处置' : '已归档')
    actionVisible.value = false
    load()
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '操作失败')
  }
}

onMounted(load)
</script>

<style scoped>
.card-header { display: flex; justify-content: space-between; align-items: center; }
</style>
